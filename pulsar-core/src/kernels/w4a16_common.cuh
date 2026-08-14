#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

// Parts shared verbatim by the two w4a16 dequant-GEMM strategies (w4a16_gemm.cu
// and w4a16_marlin.cu): activation staging, the m16n8k16 epilogue store, and the
// split-K partial reduction. Everything that a strategy tunes -- its CTA tile
// (BN), pipeline depth (STAGES) and weight layout -- stays in that strategy's own
// file; the tile geometry reaches these helpers as template parameters.

namespace pulsar {
namespace w4a16 {

// Write one output value for lane (row r, col c). The non-split path stores into
// y in y's dtype; the split path stores fp32 into this CTA's partial slice
// y_partial[kz] at base offset poff, keeping fp32 accumulate through to the
// combine kernel.
template <typename scalar_t, bool SPLIT>
__device__ __forceinline__ void
store_out(scalar_t* __restrict__ y, float* __restrict__ y_partial, int64_t poff, int r, int c, int N, float val) {
    const int64_t idx = static_cast<int64_t>(r) * N + c;
    if constexpr (SPLIT) {
        y_partial[poff + idx] = val;
    } else {
        y[idx] = static_cast<scalar_t>(val);
    }
}

// Load the x tile [BM][BK] into sX, padding out-of-range rows/cols with zero. x
// is tiny and L2-resident, so a coalesced scalar load is enough.
template <int BM, int BK, int NTHREADS, typename scalar_t>
__device__ __forceinline__ void
load_x_chunk(scalar_t* __restrict__ sX, const scalar_t* __restrict__ x, int m0, int k0, int M, int K, int tid) {
#pragma unroll
    for (int i = tid; i < BM * BK; i += NTHREADS) {
        const int r = i / BK, c = i % BK;
        const int gm = m0 + r, gk = k0 + c;
        sX[i] = (gm < M && gk < K) ? x[static_cast<int64_t>(gm) * K + gk] : static_cast<scalar_t>(0);
    }
}

// Store a warp's MROW x LSUB m16n8k16 accumulators. Fragment C[r,c] for lane L in
// each row-tile is r in {gid, gid+8}, c in {tg*2, tg*2+1} of the N subtile's
// 8-wide span, with gid = L>>2 and tg = L&3. Warp `warp` owns N subtiles
// warp + ls*NWARPS.
template <typename scalar_t, bool SPLIT, int LSUB, int MROW, int NWARPS>
__device__ __forceinline__ void store_acc_tile(
    scalar_t* __restrict__ y,
    float* __restrict__ y_partial,
    int64_t poff,
    const float (&acc)[LSUB][MROW][4],
    int m0,
    int n0,
    int M,
    int N,
    int warp,
    int lane
) {
    const int gid = lane >> 2, tg = lane & 3;
#pragma unroll
    for (int ls = 0; ls < LSUB; ++ls) {
        const int col0 = n0 + (warp + ls * NWARPS) * 8 + tg * 2;
#pragma unroll
        for (int mr = 0; mr < MROW; ++mr) {
            const int r0 = m0 + mr * 16 + gid, r1 = r0 + 8;
            const float* a = acc[ls][mr];
            if (r0 < M) {
                if (col0 + 0 < N) {
                    store_out<scalar_t, SPLIT>(y, y_partial, poff, r0, col0 + 0, N, a[0]);
                }
                if (col0 + 1 < N) {
                    store_out<scalar_t, SPLIT>(y, y_partial, poff, r0, col0 + 1, N, a[1]);
                }
            }
            if (r1 < M) {
                if (col0 + 0 < N) {
                    store_out<scalar_t, SPLIT>(y, y_partial, poff, r1, col0 + 0, N, a[2]);
                }
                if (col0 + 1 < N) {
                    store_out<scalar_t, SPLIT>(y, y_partial, poff, r1, col0 + 1, N, a[3]);
                }
            }
        }
    }
}

constexpr int kCombineThreads = 256;

// Sum the ksplits fp32 partials for each (m, n) and cast to y's dtype. y_partial
// is [ksplits, M, N] fp32 (split stride = total = M*N); one thread per output
// element. The fixed split order keeps this deterministic.
template <typename scalar_t>
__global__ void
combine_partials_kernel(scalar_t* __restrict__ y, const float* __restrict__ y_partial, int64_t total, int ksplits) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float* p = y_partial + idx;
    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < ksplits; ++i) {
        sum += p[static_cast<int64_t>(i) * total];
    }
    y[idx] = static_cast<scalar_t>(sum);
}

// Launch the combine reduction over a [ksplits, M, N] fp32 partial buffer.
template <typename scalar_t>
void launch_combine(scalar_t* y, const float* y_partial, int M, int N, int ksplits, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(M) * N;
    const int64_t blocks = (total + kCombineThreads - 1) / kCombineThreads;
    combine_partials_kernel<scalar_t>
        <<<static_cast<unsigned>(blocks), kCombineThreads, 0, stream>>>(y, y_partial, total, ksplits);
}

// K-splits for a non-split grid of base_ctas CTAs over nchunks K chunks. The
// non-split grid under-fills the SMs at decode M, so K is split to multiply it;
// past kCtaPerSm CTAs per SM the extra splits only add combine work.
constexpr int kCtaPerSm = 4;

inline int split_k_count(int64_t base_ctas, int nchunks) {
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t splits = (kCtaPerSm * static_cast<int64_t>(sm_count) + base_ctas - 1) / base_ctas;
    splits = std::min<int64_t>(std::max<int64_t>(splits, 1), nchunks);
    return static_cast<int>(splits);
}

}  // namespace w4a16
}  // namespace pulsar
