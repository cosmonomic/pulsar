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
#include <cstring>

// Marlin-style w4a16 dequant-GEMM: y = x @ dequant(W)^T, the same math as
// w4a16_gemm.cu over a weight layout repacked once at load to match the MMA
// fragment. Two ops:
//   * repack_w4a16(weight_packed, weight_scale, group_size)
//         -> (weight_marlin int32 [Npad*Kpad/8], scale_marlin fp32 [K/G, N])
//   * gemm_w4a16_marlin(x, weight_marlin, scale_marlin, group_size) -> y
// weight_packed is the compressed-tensors packing w4a16_gemm.cu documents.
//
// THE MARLIN WEIGHT LAYOUT. One int32 word holds every nibble a single lane
// consumes over a 32-wide K tile (two consecutive m16n8k16 K subchunks), so a warp
// reads a whole 8-N x 32-K tile as 32 consecutive int32 (one coalesced 128-byte
// load). The eight nibbles of lane L's word for the K tile at kbase, with
// tg = L&3 and k relative to kbase, are:
//     nibble 0 = w[n, kbase +       tg*2]      nibble 4 = w[n, kbase +       tg*2+1]
//     nibble 1 = w[n, kbase +  8  + tg*2]      nibble 5 = w[n, kbase +  8  + tg*2+1]
//     nibble 2 = w[n, kbase + 16  + tg*2]      nibble 6 = w[n, kbase + 16  + tg*2+1]
//     nibble 3 = w[n, kbase + 24  + tg*2]      nibble 7 = w[n, kbase + 24  + tg*2+1]
// so the dequant bit-trick applied to `word` (nibbles 0,4 -> b[0] and 1,5 -> b[1])
// and to `word>>8` (nibbles 2,6 and 3,7) yields the two K subchunks' B fragments
// directly, with n column = L>>2. Each stored nibble is the signed int4 value with
// its sign bit flipped (o ^ 8), because the bit-trick yields (nibble - 8).
// Out-of-range rows/columns (from padding N to BN and K to BK) store 8, i.e.
// dequant 0.
//
// Word order, CTA-tiled: column-block cb (BN cols) -> K-chunk kc (BK) -> K tile
// within chunk (BK/32) -> N subtile ns (BN/8) -> lane (32). A CTA streams each
// K-chunk as one contiguous run of (BK/32)*(BN/8)*32 int32.
//
// scale_marlin is the transpose scale_marlin[g, n] = weight_scale[n, g] (fp32), so
// the 8 N columns a warp's subtile touches at one group are 8 contiguous fp32. A
// 16-wide K subchunk lies in one group (group_size is a multiple of 16), so one
// scale multiplies all four of a lane's dequantized values.
//
// Throughput invariant, as in w4a16_gemm.cu: weight_marlin stays int4 in global
// memory (Npad*Kpad/8 int32) and is dequantized on-chip into MMA B registers; no
// bf16/fp16 weight matrix is ever written back to global memory.

namespace pulsar {
namespace {

constexpr int BN = 64;  // output N columns per CTA
constexpr int BK = 64;  // K streamed per chunk (multiple of 32)
constexpr int KTILE = 32;  // K per Marlin word (two m16n8k16 K subchunks)
constexpr int STAGES = 3;  // cp.async pipeline depth
constexpr int NWARPS = 4;
constexpr int NTHREADS = NWARPS * 32;
constexpr int NSUB = BN / 8;  // 8-wide N subtiles per CTA
constexpr int LSUB = NSUB / NWARPS;  // N subtiles owned per warp
constexpr int KSUB = BK / 16;  // 16-wide K subchunks per BK chunk
constexpr int KT_PER_CHUNK = BK / KTILE;  // Marlin K tiles per chunk
constexpr int WORDS_PER_CHUNK = KT_PER_CHUNK * NSUB * 32;  // int32 per chunk
constexpr int UINT4_PER_CHUNK = WORDS_PER_CHUNK / 4;

__host__ __device__ __forceinline__ int64_t roundup(int64_t v, int64_t m) {
    return (v + m - 1) / m * m;
}

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
    __half2 r;
    memcpy(&r, &u, 4);
    return r;
}
__device__ __forceinline__ __nv_bfloat162 as_b2(uint32_t u) {
    __nv_bfloat162 r;
    memcpy(&r, &u, 4);
    return r;
}
__device__ __forceinline__ uint32_t as_u32(__half2 h) {
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}
__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 h) {
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

// Fast int4 -> {fp16,bf16}x2 dequant (AWQ/Marlin bit-trick) with the per-group
// scale folded in. Stored nibbles are (signed value ^ 8); the trick returns
// (nibble - 8) == the signed value, then multiplies by scale. `q` packs the two
// N-subchunk halves; b[0] gets nibbles {0,4}, b[1] gets nibbles {1,5}.
template <bool BF16> __device__ __forceinline__ void dequant_scale(uint32_t b[2], uint32_t q, float scale);

template <> __device__ __forceinline__ void dequant_scale<false>(uint32_t b[2], uint32_t q, float scale) {
    // fp16: 0x6400 == 1024.0. nibble at bit0 -> 1024+n (subtract 1032); nibble
    // at bit4 -> 1024+16n (fma by 1/16, add -72) -> n-8.
    const uint32_t lo = (q & 0x000f000f) | 0x64006400;
    const uint32_t hi = (q & 0x00f000f0) | 0x64006400;
    __half2 f0 = __hsub2(as_h2(lo), as_h2(0x64086408));
    __half2 f1 = __hfma2(as_h2(hi), as_h2(0x2c002c00), as_h2(0xd480d480));
    const __half2 s2 = __half2half2(__float2half(scale));
    b[0] = as_u32(__hmul2(f0, s2));
    b[1] = as_u32(__hmul2(f1, s2));
}

template <> __device__ __forceinline__ void dequant_scale<true>(uint32_t b[2], uint32_t q, float scale) {
    // bf16: 0x4300 == 128.0. bf16's 7-bit mantissa can't hold a nibble at bit4,
    // so shift right by 4 first, then both halves are 128+n (subtract 136).
    const uint32_t lo = (q & 0x000f000f) | 0x43004300;
    const uint32_t q4 = q >> 4;
    const uint32_t hi = (q4 & 0x000f000f) | 0x43004300;
    __nv_bfloat162 f0 = __hsub2(as_b2(lo), as_b2(0x43084308));
    __nv_bfloat162 f1 = __hsub2(as_b2(hi), as_b2(0x43084308));
    const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(scale));
    b[0] = as_u32(__hmul2(f0, s2));
    b[1] = as_u32(__hmul2(f1, s2));
}

__device__ __forceinline__ int
read_nibble(const int32_t* __restrict__ wp, int n, int k, int N, int K, int pack_stride) {
    if (n >= N || k >= K) {
        return 8;  // padding -> dequant 0
    }
    const int32_t packed = wp[static_cast<int64_t>(n) * pack_stride + (k >> 3)];
    const int o = (packed >> (4 * (k & 7))) & 0xF;  // stored two's-complement 4b
    return o ^ 8;  // flip sign bit for the trick
}

// One thread per output int32 word of weight_marlin.
__global__ void
repack_weight_kernel(const int32_t* __restrict__ wp, int32_t* __restrict__ wm, int N, int K, int Npad, int Kpad) {
    const int64_t widx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(Npad) * Kpad / 8;
    if (widx >= total) {
        return;
    }

    const int nchunks = Kpad / BK;
    const int pack_stride = K / 8;

    int64_t t = widx;
    const int lane = static_cast<int>(t % 32);
    t /= 32;
    const int ns = static_cast<int>(t % NSUB);
    t /= NSUB;
    const int ktile = static_cast<int>(t % KT_PER_CHUNK);
    t /= KT_PER_CHUNK;
    const int kc = static_cast<int>(t % nchunks);
    const int cb = static_cast<int>(t / nchunks);

    const int n = (cb * NSUB + ns) * 8 + (lane >> 2);
    const int tg = lane & 3;
    const int kbase = (kc * KT_PER_CHUNK + ktile) * KTILE;

    // nibble p -> k offset within the 32-wide tile.
    const int koff[8] =
        {tg * 2, 8 + tg * 2, 16 + tg * 2, 24 + tg * 2, tg * 2 + 1, 8 + tg * 2 + 1, 16 + tg * 2 + 1, 24 + tg * 2 + 1};
    uint32_t word = 0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        const int nib = read_nibble(wp, n, kbase + koff[p], N, K, pack_stride);
        word |= static_cast<uint32_t>(nib & 0xF) << (4 * p);
    }
    wm[widx] = static_cast<int32_t>(word);
}

// scale_marlin[g, n] = weight_scale[n, g] (transpose, already fp32).
__global__ void repack_scale_kernel(const float* __restrict__ ws, float* __restrict__ sm, int N, int Kg) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<int64_t>(N) * Kg) {
        return;
    }
    const int n = static_cast<int>(idx / Kg);
    const int g = static_cast<int>(idx % Kg);
    sm[static_cast<int64_t>(g) * N + n] = ws[static_cast<int64_t>(n) * Kg + g];
}

__device__ __forceinline__ void
cpasync_marlin(int32_t* __restrict__ sMarlin, const int32_t* __restrict__ wm, int64_t base_word, int tid) {
#pragma unroll
    for (int u = tid; u < UINT4_PER_CHUNK; u += NTHREADS) {
        void* dst = &sMarlin[u * 4];
        const void* src = &wm[base_word + static_cast<int64_t>(u) * 4];
        __pipeline_memcpy_async(dst, src, sizeof(uint4));
    }
}

template <int BM, typename scalar_t, bool BF16, bool SPLIT>
__global__ void __launch_bounds__(NTHREADS) w4a16_marlin_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ wm,
    const float* __restrict__ sm,
    scalar_t* __restrict__ y,
    float* __restrict__ y_partial,
    int M,
    int N,
    int K,
    int Kpad,
    int group_size,
    int chunks_per_split
) {
    constexpr int MROW = BM / 16;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cb = blockIdx.x;  // column-block index
    const int n0 = cb * BN;
    const int m0 = blockIdx.y * BM;

    const int nchunks = Kpad / BK;
    const int kz = SPLIT ? static_cast<int>(blockIdx.z) : 0;
    const int chunk_begin = SPLIT ? kz * chunks_per_split : 0;
    const int chunk_end = SPLIT ? min((kz + 1) * chunks_per_split, nchunks) : nchunks;
    const int local_nchunks = chunk_end - chunk_begin;

    __shared__ scalar_t sX[STAGES][BM * BK];
    __shared__ int32_t sMarlin[STAGES][WORDS_PER_CHUNK];

    const int gid = lane >> 2;
    const int quad = lane >> 3, rin = lane & 7;
    const int arow = (quad & 1) * 8 + rin, acol = (quad >> 1) * 8;

    float acc[LSUB][MROW][4];
#pragma unroll
    for (int ls = 0; ls < LSUB; ++ls)
#pragma unroll
        for (int mr = 0; mr < MROW; ++mr) {
            acc[ls][mr][0] = acc[ls][mr][1] = acc[ls][mr][2] = acc[ls][mr][3] = 0.0f;
        }

    // First int32 word of column-block cb, chunk c: contiguous run per chunk.
    auto chunk_base = [&](int c) -> int64_t {
        return (static_cast<int64_t>(cb) * nchunks + c) * (static_cast<int64_t>(KT_PER_CHUNK) * NSUB * 32);
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < local_nchunks) {
            const int c = chunk_begin + s;
            cpasync_marlin(sMarlin[s], wm, chunk_base(c), tid);
            w4a16::load_x_chunk<BM, BK, NTHREADS>(sX[s], x, m0, c * BK, M, K, tid);
        }
        __pipeline_commit();
    }

    for (int lc = 0; lc < local_nchunks; ++lc) {
        const int c = chunk_begin + lc;
        const int buf = lc % STAGES;
        const int k0 = c * BK;
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();
        const int fetch = lc + STAGES - 1;
        if (fetch < local_nchunks) {
            const int fc = chunk_begin + fetch;
            const int fbuf = fetch % STAGES;
            cpasync_marlin(sMarlin[fbuf], wm, chunk_base(fc), tid);
            w4a16::load_x_chunk<BM, BK, NTHREADS>(sX[fbuf], x, m0, fc * BK, M, K, tid);
        }
        __pipeline_commit();

#pragma unroll
        for (int kc16 = 0; kc16 < KSUB; ++kc16) {
            const int ktile = kc16 >> 1;
            const int s = kc16 & 1;
            uint32_t ra[MROW][4];
#pragma unroll
            for (int mr = 0; mr < MROW; ++mr) {
                mma::ldmatrix_x4(ra[mr], &sX[buf][(mr * 16 + arow) * BK + kc16 * 16 + acol]);
            }
#pragma unroll
            for (int ls = 0; ls < LSUB; ++ls) {
                const int ns = warp + ls * NWARPS;
                uint32_t q = sMarlin[buf][(ktile * NSUB + ns) * 32 + lane];
                if (s) {
                    q >>= 8;
                }
                const int ncol = n0 + ns * 8 + gid;
                const int k16start = k0 + ktile * KTILE + s * 16;
                float sc = 0.0f;
                if (ncol < N && k16start < K) {
                    sc = sm[static_cast<int64_t>(k16start / group_size) * N + ncol];
                }
                uint32_t b[2];
                dequant_scale<BF16>(b, q, sc);
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
    const int32_t* wm,
    const float* sm,
    scalar_t* y,
    float* yp,
    int M,
    int N,
    int K,
    int Kpad,
    int group_size,
    int cps
) {
    w4a16_marlin_kernel<BM, scalar_t, BF16, SPLIT>
        <<<grid, NTHREADS, 0, stream>>>(x, wm, sm, y, yp, M, N, K, Kpad, group_size, cps);
}

template <int BM, typename scalar_t, bool BF16>
void dispatch(
    bool split,
    const dim3& grid,
    cudaStream_t stream,
    const scalar_t* x,
    const int32_t* wm,
    const float* sm,
    scalar_t* y,
    float* yp,
    int M,
    int N,
    int K,
    int Kpad,
    int group_size,
    int cps,
    int ksplits
) {
    if (split) {
        launch<BM, scalar_t, BF16, true>(grid, stream, x, wm, sm, y, yp, M, N, K, Kpad, group_size, cps);
        w4a16::launch_combine(y, yp, M, N, ksplits, stream);
    } else {
        launch<BM, scalar_t, BF16, false>(grid, stream, x, wm, sm, y, nullptr, M, N, K, Kpad, group_size, cps);
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor>
repack_w4a16_cuda(const at::Tensor& weight_packed, const at::Tensor& weight_scale, int64_t group_size) {
    TORCH_CHECK(weight_packed.is_cuda() && weight_scale.is_cuda(), "repack_w4a16: inputs must be CUDA tensors");
    TORCH_CHECK(weight_packed.scalar_type() == at::kInt, "repack_w4a16: weight_packed must be int32");
    TORCH_CHECK(
        weight_packed.dim() == 2 && weight_scale.dim() == 2,
        "repack_w4a16: weight_packed [N,K/8] and weight_scale [N,K/G]"
    );
    TORCH_CHECK(group_size > 0 && group_size % 16 == 0, "repack_w4a16: group_size must be a positive multiple of 16");

    const int64_t N = weight_packed.size(0);
    const int64_t K = weight_packed.size(1) * 8;
    TORCH_CHECK(K % group_size == 0, "repack_w4a16: K must be a multiple of group_size");
    const int64_t Kg = K / group_size;
    TORCH_CHECK(
        weight_scale.size(0) == N && weight_scale.size(1) == Kg,
        "repack_w4a16: weight_scale must be [N, K/group_size]"
    );

    const int64_t Npad = roundup(N, BN);
    const int64_t Kpad = roundup(K, BK);
    const int64_t total = Npad * Kpad / 8;

    auto wp = weight_packed.contiguous();
    auto ws = weight_scale.to(at::kFloat).contiguous();
    auto weight_marlin = at::empty({total}, wp.options());
    auto scale_marlin = at::empty({Kg, N}, ws.options());

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    if (total > 0) {
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
        repack_weight_kernel<<<blocks, threads, 0, stream>>>(
            wp.data_ptr<int32_t>(),
            weight_marlin.data_ptr<int32_t>(),
            static_cast<int>(N),
            static_cast<int>(K),
            static_cast<int>(Npad),
            static_cast<int>(Kpad)
        );
    }
    const int64_t sn = N * Kg;
    if (sn > 0) {
        const unsigned sblocks = static_cast<unsigned>((sn + threads - 1) / threads);
        repack_scale_kernel<<<sblocks, threads, 0, stream>>>(
            ws.data_ptr<float>(),
            scale_marlin.data_ptr<float>(),
            static_cast<int>(N),
            static_cast<int>(Kg)
        );
    }
    return {weight_marlin, scale_marlin};
}

at::Tensor gemm_w4a16_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t group_size
) {
    TORCH_CHECK(
        x.is_cuda() && weight_marlin.is_cuda() && scale_marlin.is_cuda(),
        "gemm_w4a16_marlin: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
        "gemm_w4a16_marlin: x must be fp16 or bf16"
    );
    TORCH_CHECK(weight_marlin.scalar_type() == at::kInt, "gemm_w4a16_marlin: weight_marlin must be int32");
    TORCH_CHECK(scale_marlin.scalar_type() == at::kFloat, "gemm_w4a16_marlin: scale_marlin must be fp32");
    TORCH_CHECK(x.dim() == 2, "gemm_w4a16_marlin: x must be 2-D [M, K]");
    TORCH_CHECK(scale_marlin.dim() == 2, "gemm_w4a16_marlin: scale_marlin must be 2-D [K/G, N]");
    TORCH_CHECK(
        group_size > 0 && group_size % 16 == 0,
        "gemm_w4a16_marlin: group_size must be a positive multiple of 16"
    );
    TORCH_CHECK(x.is_contiguous(), "gemm_w4a16_marlin: x must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = scale_marlin.size(1);
    const int64_t Kg = scale_marlin.size(0);
    TORCH_CHECK(Kg == K / group_size && K % group_size == 0, "gemm_w4a16_marlin: scale_marlin K/G must match x's K");
    const int64_t Kpad = roundup(K, BK);
    const int64_t Npad = roundup(N, BN);
    TORCH_CHECK(
        weight_marlin.numel() == Npad * Kpad / 8,
        "gemm_w4a16_marlin: weight_marlin has wrong size for this shape"
    );

    auto xc = x.contiguous();
    auto wm = weight_marlin.contiguous();
    auto sm = scale_marlin.contiguous();
    auto y = at::empty({M, N}, xc.options());
    if (M == 0 || N == 0) {
        return y;
    }

    const int bm = (M <= 16) ? 16 : 64;
    const int nchunks = static_cast<int>(Kpad / BK);
    const unsigned gx = static_cast<unsigned>(Npad / BN);
    const unsigned gy = static_cast<unsigned>((M + bm - 1) / bm);

    const int ksplits = w4a16::split_k_count(static_cast<int64_t>(gx) * gy, nchunks);
    const bool split = ksplits > 1;
    const int cps = split ? (nchunks + ksplits - 1) / ksplits : nchunks;
    const dim3 grid(gx, gy, split ? static_cast<unsigned>(ksplits) : 1u);
    auto stream = at::cuda::getCurrentCUDAStream();

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
                wm.data_ptr<int32_t>(),
                sm.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)Kpad,
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
                wm.data_ptr<int32_t>(),
                sm.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)Kpad,
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
                wm.data_ptr<int32_t>(),
                sm.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)Kpad,
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
                wm.data_ptr<int32_t>(),
                sm.data_ptr<float>(),
                yout,
                ypp,
                (int)M,
                (int)N,
                (int)K,
                (int)Kpad,
                (int)group_size,
                cps,
                ksplits
            );
        }
    }
    return y;
}

}  // namespace pulsar
