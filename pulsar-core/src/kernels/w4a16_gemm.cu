#include "w4a16_common.cuh"

#include "pulsar/ops.hpp"

#include "pulsar/kernels/mma.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cstdint>

// Fused w4a16 dequant-GEMM: y = x @ dequant(W)^T, i.e.
//   y[m,n] = sum_k x[m,k] * w_deq[n,k]
// for a Linear with int4 group-wise symmetric weights.
//
//   x             [M, K]      fp16/bf16, row-major, contiguous (A operand).
//   weight_packed [N, K/8]    int32; 8 signed int4 per int32 along K, LSB-first
//                             (column k in nibble k%8 of weight_packed[n, k/8]),
//                             two's-complement -> [-8, 7].
//   weight_scale  [N, K/G]    fp32 (host-upcast); w_deq[n,k] = q[n,k]*scale[n,k/G].
//   y             [M, N]      same dtype as x.
//
// Throughput invariant: weights are read from global memory as packed int4 and
// dequantized on-chip into MMA B-operand registers. The dequantized weight matrix
// is NEVER written back to global memory, and there is no bf16/fp16 weight tile in
// shared memory.
//
// The inner op is the m16n8k16 tensor-core MMA (A = x, B = the dequantized weight
// fragment) with fp32 accumulate.

namespace pulsar {
namespace {

constexpr int BN = 32;  // output N columns per CTA
constexpr int BK = 64;  // K streamed per chunk (multiple of 16)
constexpr int STAGES = 2;  // cp.async pipeline depth (chunks in flight per CTA)
constexpr int NWARPS = 4;  // warps per CTA
constexpr int NTHREADS = NWARPS * 32;
constexpr int NSUB = BN / 8;  // 8-wide N subtiles
constexpr int LSUB = NSUB / NWARPS;  // N subtiles owned per warp
constexpr int KSUB = BK / 16;  // 16-wide K subchunks per BK chunk
constexpr int PACK_PER_ROW = BK / 8;  // int32 columns per weight row per chunk
constexpr int UINT4_PER_ROW = PACK_PER_ROW / 4;  // 128-bit loads per row per chunk
constexpr int PACK_INT32 = BN * PACK_PER_ROW;  // packed int32 per weight chunk
constexpr int TU = BN * UINT4_PER_ROW;  // 128-bit weight loads per chunk

// Extract signed int4 nibble j (j in [0,8)) from a packed int32 -> [-8, 7].
__device__ __forceinline__ int unpack_nibble(int32_t packed, int j) {
    int nib = (packed >> (4 * j)) & 0xF;
    return nib - ((nib >> 3) << 4);  // sign-extend: nib>=8 -> nib-16
}

// Async-copy one BN x BK packed weight chunk (as uint4) from global into the
// shared staging buffer sPack. Threads own a fixed uint4 slot (u -> sPack[u*4]),
// the SAME slot they dequant below, so no barrier is needed between the load and
// the dequant -- only the per-thread cp.async wait. Out-of-range rows/cols are
// zero-filled with a zeroing cp.async (no source bytes copied).
__device__ __forceinline__ void cpasync_pack_chunk(
    int32_t* __restrict__ sPack,
    const int32_t* __restrict__ weight_packed,
    int n0,
    int k0,
    int N,
    int K,
    int pack_stride,
    int tid
) {
#pragma unroll
    for (int u = tid; u < TU; u += NTHREADS) {
        const int nl = u / UINT4_PER_ROW;
        const int uc = u % UINT4_PER_ROW;  // uint4 column within chunk
        const int gn = n0 + nl;
        const int gk8 = k0 / 8 + uc * 4;  // int32 column in weight_packed
        void* dst = &sPack[u * 4];
        const bool ok = (gn < N) && (k0 + uc * 32) < K;
        if (ok) {
            const void* src = &weight_packed[static_cast<int64_t>(gn) * pack_stride + gk8];
            __pipeline_memcpy_async(dst, src, sizeof(uint4));
        } else {
            __pipeline_memcpy_async(dst, dst, sizeof(uint4), sizeof(uint4));
        }
    }
}

// Pack two fp32 values into a b16x2 register (low half = a, high half = b), in
// the MMA operand dtype (bf16 if BF16, else fp16).
template <bool BF16> __device__ __forceinline__ uint32_t pack2(float a, float b) {
    if constexpr (BF16) {
        return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(a))) |
            (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16(b))) << 16);
    } else {
        return static_cast<uint32_t>(__half_as_ushort(__float2half(a))) |
            (static_cast<uint32_t>(__half_as_ushort(__float2half(b))) << 16);
    }
}

// Build the m16n8k16 B fragment (weights, .col) for this lane directly from the
// packed int4 chunk in shared memory -- no bf16 weight tile, no ldmatrix. For an
// 8-wide N subtile at row nl_base and a 16-wide K subchunk kc, this lane owns
// N column (nl_base + lane>>2) and, per the .col B layout, K rows {tg*2, tg*2+1}
// (register b0) and {8+tg*2, 8+tg*2+1} (register b1) with tg = lane&3. The whole
// 16-wide K subchunk lies in one quant group (group_size % 16 == 0, kc*16 is
// 16-aligned), so a single scale multiplies all four values.
template <bool BF16>
__device__ __forceinline__ void bfrag(
    uint32_t b[2],
    const int32_t* __restrict__ sPack,
    const float* __restrict__ weight_scale,
    int nl_base,
    int lane,
    int n0,
    int k0,
    int kc,
    int N,
    int K,
    int group_size,
    int scale_stride
) {
    const int nl = nl_base + (lane >> 2);
    const int tg = lane & 3;
    const int gn = n0 + nl;
    const int k16 = k0 + kc * 16;
    if (gn >= N || k16 >= K) {
        b[0] = 0;
        b[1] = 0;
        return;
    }
    const int32_t p0 = sPack[nl * PACK_PER_ROW + kc * 2];  // K rows 0..7
    const int32_t p1 = sPack[nl * PACK_PER_ROW + kc * 2 + 1];  // K rows 8..15
    const float scale = weight_scale[static_cast<int64_t>(gn) * scale_stride + k16 / group_size];
    b[0] = pack2<BF16>(unpack_nibble(p0, tg * 2) * scale, unpack_nibble(p0, tg * 2 + 1) * scale);
    b[1] = pack2<BF16>(unpack_nibble(p1, tg * 2) * scale, unpack_nibble(p1, tg * 2 + 1) * scale);
}

// One CTA computes a BM x BN output tile (BM = MROW MMA row-tiles). Shared memory
// holds only the packed int4 weight chunk (sPack) and the x tile (sX), both
// cp.async double-buffered: the next K-chunk is prefetched while the current
// chunk's MMAs run. Each weight byte is read from global exactly once and reused
// across all MROW row-tiles.
//
// Split-K (SPLIT=true): blockIdx.z = kz selects the K-slice. The K chunks are
// partitioned into contiguous runs of chunks_per_split; split kz owns chunks
// [kz*cps, min((kz+1)*cps, nchunks)). Each CTA accumulates only its slice's
// partial sum and writes it (fp32) into y_partial[kz]; the combine kernel sums the
// ksplits partials into y. An empty split (kz*cps >= nchunks) runs no MMAs and
// writes its zero-initialized accumulator, contributing 0 to the reduction.
template <int BM, typename scalar_t, bool BF16, bool SPLIT>
__global__ void __launch_bounds__(NTHREADS) gemm_w4a16_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ weight_packed,
    const float* __restrict__ weight_scale,
    scalar_t* __restrict__ y,
    float* __restrict__ y_partial,
    int M,
    int N,
    int K,
    int group_size,
    int chunks_per_split
) {
    constexpr int MROW = BM / 16;  // MMA row-tiles stacked in the M dimension
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int n0 = blockIdx.x * BN;
    const int m0 = blockIdx.y * BM;

    const int nchunks = (K + BK - 1) / BK;
    // K-chunk range [chunk_begin, chunk_end) this CTA owns. Non-split: the whole
    // range. Split: this CTA's contiguous slice (may be empty for a high kz).
    const int kz = SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int chunk_begin = SPLIT ? kz * chunks_per_split : 0;
    const int chunk_end = SPLIT ? min((kz + 1) * chunks_per_split, nchunks) : nchunks;
    const int local_nchunks = chunk_end - chunk_begin;  // may be <= 0 (empty)

    __shared__ scalar_t sX[STAGES][BM * BK];
    __shared__ int32_t sPack[STAGES][PACK_INT32];  // multi-stage cp.async target

    const int quad = lane >> 3, rin = lane & 7;
    const int arow = (quad & 1) * 8 + rin, acol = (quad >> 1) * 8;

    float acc[LSUB][MROW][4];
#pragma unroll
    for (int ls = 0; ls < LSUB; ++ls)
#pragma unroll
        for (int mr = 0; mr < MROW; ++mr) {
            acc[ls][mr][0] = acc[ls][mr][1] = acc[ls][mr][2] = acc[ls][mr][3] = 0.0f;
        }

    const int pack_stride = K / 8;
    const int scale_stride = K / group_size;

    // Prologue: issue the first STAGES-1 chunk loads so several cp.async groups
    // are in flight at once.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < local_nchunks) {
            const int k0 = (chunk_begin + s) * BK;
            cpasync_pack_chunk(sPack[s], weight_packed, n0, k0, N, K, pack_stride, tid);
            w4a16::load_x_chunk<BM, BK, NTHREADS>(sX[s], x, m0, k0, M, K, tid);
        }
        __pipeline_commit();
    }

    // Steady state: wait until the current chunk's group is the oldest still
    // outstanding, run its MMAs while STAGES-2 later chunks keep loading, and
    // issue the load for the chunk STAGES-1 ahead. One __syncthreads/iter.
    for (int lc = 0; lc < local_nchunks; ++lc) {
        const int c = chunk_begin + lc;
        const int buf = lc % STAGES;
        const int k0 = c * BK;
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        const int fetch = lc + STAGES - 1;
        if (fetch < local_nchunks) {
            const int fbuf = fetch % STAGES;
            const int k_f = (chunk_begin + fetch) * BK;
            cpasync_pack_chunk(sPack[fbuf], weight_packed, n0, k_f, N, K, pack_stride, tid);
            w4a16::load_x_chunk<BM, BK, NTHREADS>(sX[fbuf], x, m0, k_f, M, K, tid);
        }
        __pipeline_commit();

        // A fragments depend only on (mr, kc), not the N-subtile, so load them
        // once per kc and reuse across all LSUB N-subtiles.
#pragma unroll
        for (int kc = 0; kc < KSUB; ++kc) {
            uint32_t ra[MROW][4];
#pragma unroll
            for (int mr = 0; mr < MROW; ++mr) {
                mma::ldmatrix_x4(ra[mr], &sX[buf][(mr * 16 + arow) * BK + kc * 16 + acol]);
            }
#pragma unroll
            for (int ls = 0; ls < LSUB; ++ls) {
                uint32_t b[2];
                bfrag<BF16>(
                    b,
                    sPack[buf],
                    weight_scale,
                    (warp + ls * NWARPS) * 8,
                    lane,
                    n0,
                    k0,
                    kc,
                    N,
                    K,
                    group_size,
                    scale_stride
                );
#pragma unroll
                for (int mr = 0; mr < MROW; ++mr) {
                    mma::mma_m16n8k16<BF16>(acc[ls][mr], ra[mr], b, acc[ls][mr]);
                }
            }
        }
    }

    const int64_t poff = static_cast<int64_t>(kz) * M * N;
    w4a16::store_acc_tile<scalar_t, SPLIT, LSUB, MROW, NWARPS>(y, y_partial, poff, acc, m0, n0, M, N, warp, lane);
}

template <int BM, typename scalar_t, bool BF16, bool SPLIT>
void launch(
    const dim3& grid,
    cudaStream_t stream,
    const scalar_t* x,
    const int32_t* wp,
    const float* ws,
    scalar_t* y,
    float* yp,
    int M,
    int N,
    int K,
    int group_size,
    int cps
) {
    gemm_w4a16_kernel<BM, scalar_t, BF16, SPLIT>
        <<<grid, NTHREADS, 0, stream>>>(x, wp, ws, y, yp, M, N, K, group_size, cps);
}

// Dispatch the split (grid.z = ksplits, fp32 partials) or non-split (grid.z = 1,
// direct y) kernel for a given (bm, dtype), then, when splitting, the combine.
template <int BM, typename scalar_t, bool BF16>
void dispatch(
    bool split,
    const dim3& grid,
    cudaStream_t stream,
    const scalar_t* x,
    const int32_t* wp,
    const float* ws,
    scalar_t* y,
    float* yp,
    int M,
    int N,
    int K,
    int group_size,
    int cps,
    int ksplits
) {
    if (split) {
        launch<BM, scalar_t, BF16, true>(grid, stream, x, wp, ws, y, yp, M, N, K, group_size, cps);
        w4a16::launch_combine(y, yp, M, N, ksplits, stream);
    } else {
        launch<BM, scalar_t, BF16, false>(grid, stream, x, wp, ws, y, nullptr, M, N, K, group_size, cps);
    }
}

// Shared implementation for both the auto op and the forced-split test op.
// forced_splits <= 0 => pick ksplits from the occupancy heuristic and use the
// direct (non-split) kernel when it comes out 1. forced_splits >= 1 => always
// route through the split+combine path with exactly that many splits (allowed to
// exceed nchunks; the extra splits are empty and contribute 0).
at::Tensor gemm_w4a16_run(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int forced_splits
) {
    TORCH_CHECK(
        x.is_cuda() && weight_packed.is_cuda() && weight_scale.is_cuda(),
        "gemm_w4a16: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "gemm_w4a16: x must be fp16 or bf16");
    TORCH_CHECK(weight_packed.scalar_type() == at::kInt, "gemm_w4a16: weight_packed must be int32");
    TORCH_CHECK(x.dim() == 2, "gemm_w4a16: x must be 2-D [M, K]");
    TORCH_CHECK(weight_packed.dim() == 2, "gemm_w4a16: weight_packed must be 2-D [N, K/8]");
    TORCH_CHECK(weight_scale.dim() == 2, "gemm_w4a16: weight_scale must be 2-D [N, K/group_size]");
    TORCH_CHECK(x.is_contiguous(), "gemm_w4a16: x must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = weight_packed.size(0);

    TORCH_CHECK(group_size > 0 && group_size % 16 == 0, "gemm_w4a16: group_size must be a positive multiple of 16");
    TORCH_CHECK(
        K % 8 == 0 && weight_packed.size(1) == K / 8,
        "gemm_w4a16: weight_packed must be [N, K/8] with K a multiple of 8"
    );
    TORCH_CHECK(K % group_size == 0, "gemm_w4a16: K must be a multiple of group_size");
    TORCH_CHECK(
        weight_scale.size(0) == N && weight_scale.size(1) == K / group_size,
        "gemm_w4a16: weight_scale must be [N, K/group_size]"
    );

    auto xc = x.contiguous();
    auto wp = weight_packed.contiguous();
    auto ws = weight_scale.to(at::kFloat).contiguous();
    auto y = at::empty({M, N}, xc.options());

    if (M == 0 || N == 0) {
        return y;
    }

    // BM = 16 for the decode M (grid.y == 1); BM = 64 stacks four MMA row-tiles
    // so a single CTA covers up to 64 rows without re-reading the weight tile.
    const int bm = (M <= 16) ? 16 : 64;
    const int nchunks = static_cast<int>((K + BK - 1) / BK);
    const unsigned gx = static_cast<unsigned>((N + BN - 1) / BN);
    const unsigned gy = static_cast<unsigned>((M + bm - 1) / bm);

    // forced_splits >= 1 forces the split path (even at 1, and may exceed
    // nchunks); otherwise ksplits == 1 uses the direct kernel.
    int ksplits;
    bool split;
    if (forced_splits >= 1) {
        ksplits = forced_splits;
        split = true;
    } else {
        ksplits = w4a16::split_k_count(static_cast<int64_t>(gx) * gy, nchunks);
        split = ksplits > 1;
    }
    // chunks_per_split partitions the nchunks K-chunks into contiguous runs.
    const int cps = split ? (nchunks + ksplits - 1) / ksplits : nchunks;
    const dim3 grid(gx, gy, split ? static_cast<unsigned>(ksplits) : 1u);
    auto stream = at::cuda::getCurrentCUDAStream();

    // fp32 partials [ksplits, M, N] for the deterministic combine reduction.
    at::Tensor yp32;
    if (split) {
        yp32 = at::empty({static_cast<int64_t>(ksplits), M, N}, xc.options().dtype(at::kFloat));
    }
    float* ypp = split ? yp32.data_ptr<float>() : nullptr;

    if (x.scalar_type() == at::kBFloat16) {
        auto* xp = xc.data_ptr<at::BFloat16>();
        auto* yout = y.data_ptr<at::BFloat16>();
        if (bm == 16) {
            dispatch<16, at::BFloat16, true>(
                split,
                grid,
                stream,
                xp,
                wp.data_ptr<int32_t>(),
                ws.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)group_size,
                cps,
                ksplits
            );
        } else {
            dispatch<64, at::BFloat16, true>(
                split,
                grid,
                stream,
                xp,
                wp.data_ptr<int32_t>(),
                ws.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)group_size,
                cps,
                ksplits
            );
        }
    } else {
        auto* xp = xc.data_ptr<at::Half>();
        auto* yout = y.data_ptr<at::Half>();
        if (bm == 16) {
            dispatch<16, at::Half, false>(
                split,
                grid,
                stream,
                xp,
                wp.data_ptr<int32_t>(),
                ws.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)group_size,
                cps,
                ksplits
            );
        } else {
            dispatch<64, at::Half, false>(
                split,
                grid,
                stream,
                xp,
                wp.data_ptr<int32_t>(),
                ws.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)group_size,
                cps,
                ksplits
            );
        }
    }
    return y;
}

}  // namespace

at::Tensor gemm_w4a16_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size
) {
    return gemm_w4a16_run(x, weight_packed, weight_scale, group_size, 0);
}

at::Tensor gemm_w4a16_split_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int64_t num_splits
) {
    TORCH_CHECK(num_splits >= 1, "gemm_w4a16_split: num_splits must be >= 1");
    return gemm_w4a16_run(x, weight_packed, weight_scale, group_size, static_cast<int>(num_splits));
}

}  // namespace pulsar
