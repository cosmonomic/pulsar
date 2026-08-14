#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "pulsar/kernels/mma.cuh"

// Shared m16n8k16 tensor-core attention tiles for the sm_120 paged-attention
// kernels (decode + prefill). Both GEMMs run on the bf16/fp16 MMA in mma.cuh
// with fp32 accumulate.
//
// Fragment/tile layout invariants (do not reorder the register pairings; they
// encode the m16n8k16 operand layout):
//   * MMA C/A/B operand element ownership for lane L: gid=L>>2, tg=L&3.
//     C[r,c]: r in {gid, gid+8}, c in {tg*2, tg*2+1}.
//   * ldmatrix.x4 loads a 16x16 shared tile as 4 quadrants ordered
//     (rows0-7,cols0-7),(rows8-15,cols0-7),(rows0-7,cols8-15),(rows8-15,cols8-15).
//     Lane L addresses row (L&7) of quadrant (L>>3): row=((quad&1)*8+rin),
//     col-base=((quad>>1)*8).
//   * QKᵀ: A=Q (non-trans), B=K (non-trans, K stored [key][d]); the 16-wide K
//     contraction for keys 0-7 is regs {kf0,kf2}, keys 8-15 is {kf1,kf3}.
//   * PV: A=P (non-trans), B=V loaded ldmatrix.trans (V stored [key][dim]); the
//     dim N-slice 0-7 is regs {vb0,vb1}, dim 8-15 is {vb2,vb3}.
//   * All ldmatrix row addresses are 8-element (16-byte) aligned: HEAD_DIM and
//     KEY_TILE are multiples of 16, the col base is 0 or 8, and the caller's
//     tiles are declared __align__(16).
//
// KEY_TILE is the key-tile width in keys. Decode streams one page per tile, so
// there it equals the page size; prefill may stream several pages per tile.

namespace pulsar {
namespace attn {

// Asynchronous 16-byte global->shared staging. The copy takes no register and no
// store instruction of its own, so a warp that issues it runs on until it waits.
//
// Both addresses must be 16-byte aligned. src_bytes below 16 zero-fills the
// remainder, which is how a key page's tail past the end of a sequence is cleared.
// Ordering is by committed GROUP: cp_async_wait<N> returns once at most N of the
// most recently committed groups are still in flight, and every thread must commit
// the same number of groups for that count to mean the same thing across the CTA.
// A wait covers only the calling thread's copies, so a __syncthreads() still has to
// follow before another thread reads what it staged.
__device__ __forceinline__ void cp_async_16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(mma::cvta(dst)), "l"(src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Bank-conflict-free column permutation for a shared tile of ROW_ELEMS 2-byte
// elements per row. A row is ROW_ELEMS/8 chunks of 16 bytes and an ldmatrix
// quadrant addresses 8 rows at one chunk index; with a plain row-major layout
// every row of the tile starts on the same bank and all 8 addresses collide.
// XOR-ing the chunk index with the row's low 3 bits sends them to 8 different
// chunk slots of the same 128-byte bank period. Both the staging store and the
// ldmatrix load must apply it. ROW_ELEMS must be a multiple of 64 so the
// permuted chunk stays inside the row.
template <int ROW_ELEMS> __device__ __forceinline__ int swizzled_offset(int row, int col) {
    static_assert(ROW_ELEMS % 64 == 0, "the swizzle needs at least 8 chunks per row");
    constexpr int kChunkElems = 8;
    return row * ROW_ELEMS + ((col / kChunkElems) ^ (row & 7)) * kChunkElems + col % kChunkElems;
}

// Offset of (row, col) in a tile of ROW_ELEMS-element rows, permuted or not.
template <int ROW_ELEMS, bool SWIZZLE> __device__ __forceinline__ int tile_offset(int row, int col) {
    if constexpr (SWIZZLE) {
        return swizzled_offset<ROW_ELEMS>(row, col);
    }
    return row * ROW_ELEMS + col;
}

// QKᵀ over the loaded tile: sS[m][n] = sum_d sQ[m][d] * sK[n][d] (raw, unscaled).
// The NSUB 8-key n-subtiles are striped across the CTA's warps. SCORE_STRIDE is
// sS's row stride: sS is written and read by scalar accesses only, so its rows may
// be padded past KEY_TILE to spread a per-row sweep over the shared banks. SWIZZLE
// states that the caller staged sQ and sK through swizzled_offset.
template <typename scalar_t, int KEY_TILE, int HEAD_DIM, bool BF16, int SCORE_STRIDE = KEY_TILE, bool SWIZZLE = false>
__device__ __forceinline__ void tile_qkt(const scalar_t* sQ, const scalar_t* sK, float* sS, int warp, int nwarps) {
    constexpr int DC = HEAD_DIM / 16;  // d-chunks
    constexpr int NSUB = KEY_TILE / 8;  // 8-key n-subtiles

    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int quad = lane >> 3, rin = lane & 7;
    const int arow = (quad & 1) * 8 + rin, acol = (quad >> 1) * 8;

    for (int ns = warp; ns < NSUB; ns += nwarps) {
        const int kc = ns >> 1;  // 16-key chunk
        const int within = ns & 1;  // low/high 8 keys of the chunk
        float acc[4] = {0, 0, 0, 0};
#pragma unroll
        for (int dc = 0; dc < DC; ++dc) {
            uint32_t ra[4], kf[4];
            mma::ldmatrix_x4(ra, &sQ[tile_offset<HEAD_DIM, SWIZZLE>(arow, dc * 16 + acol)]);
            mma::ldmatrix_x4(kf, &sK[tile_offset<HEAD_DIM, SWIZZLE>(kc * 16 + arow, dc * 16 + acol)]);
            uint32_t b[2] = {within ? kf[1] : kf[0], within ? kf[3] : kf[2]};
            mma::mma_m16n8k16<BF16>(acc, ra, b, acc);
        }
        const int cb = ns * 8;
        sS[(gid)*SCORE_STRIDE + cb + tg * 2 + 0] = acc[0];
        sS[(gid)*SCORE_STRIDE + cb + tg * 2 + 1] = acc[1];
        sS[(gid + 8) * SCORE_STRIDE + cb + tg * 2 + 0] = acc[2];
        sS[(gid + 8) * SCORE_STRIDE + cb + tg * 2 + 1] = acc[3];
    }
}

// PV over the loaded tile: sO[m][d] += sum_n sP[m][n] * sV[n][d]. The DSLAB
// 16-wide dim slabs are striped across the CTA's warps (disjoint sO columns).
template <typename scalar_t, int KEY_TILE, int HEAD_DIM, bool BF16>
__device__ __forceinline__ void tile_pv(const scalar_t* sP, const scalar_t* sV, float* sO, int warp, int nwarps) {
    constexpr int KC = KEY_TILE / 16;
    constexpr int DSLAB = HEAD_DIM / 16;

    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, tg = lane & 3;
    const int quad = lane >> 3, rin = lane & 7;
    const int arow = (quad & 1) * 8 + rin, acol = (quad >> 1) * 8;

    uint32_t pa[KC][4];
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
        mma::ldmatrix_x4(pa[kc], &sP[arow * KEY_TILE + kc * 16 + acol]);
    }

    for (int ds = warp; ds < DSLAB; ds += nwarps) {
        float of0[4] = {0, 0, 0, 0}, of1[4] = {0, 0, 0, 0};
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            uint32_t vb[4];
            mma::ldmatrix_x4_trans(vb, &sV[(kc * 16 + arow) * HEAD_DIM + ds * 16 + acol]);
            uint32_t b0[2] = {vb[0], vb[1]};  // dims ds*16+0..7
            uint32_t b1[2] = {vb[2], vb[3]};  // dims ds*16+8..15
            mma::mma_m16n8k16<BF16>(of0, pa[kc], b0, of0);
            mma::mma_m16n8k16<BF16>(of1, pa[kc], b1, of1);
        }
        const int cb = ds * 16;
        sO[(gid)*HEAD_DIM + cb + tg * 2 + 0] += of0[0];
        sO[(gid)*HEAD_DIM + cb + tg * 2 + 1] += of0[1];
        sO[(gid + 8) * HEAD_DIM + cb + tg * 2 + 0] += of0[2];
        sO[(gid + 8) * HEAD_DIM + cb + tg * 2 + 1] += of0[3];
        sO[(gid)*HEAD_DIM + cb + 8 + tg * 2 + 0] += of1[0];
        sO[(gid)*HEAD_DIM + cb + 8 + tg * 2 + 1] += of1[1];
        sO[(gid + 8) * HEAD_DIM + cb + 8 + tg * 2 + 0] += of1[2];
        sO[(gid + 8) * HEAD_DIM + cb + 8 + tg * 2 + 1] += of1[3];
    }
}

// PV over the loaded tile with the running output held in registers instead of a
// shared tile: out = out * row_rescale[row] + sum_n sP[m][n] * sV[n][d]. Warp w
// owns dim slabs w, w + NWARPS, ...; `out` holds kPvSlabAccums entries per slab,
// slabs in ascending ds. Entry i of the slab at column base cb = ds*16 is, for
// gid = lane>>2 and tg = lane&3, the (row, col) of the output tile
//   0:(gid, cb+tg*2)      1:(gid, cb+tg*2+1)
//   2:(gid+8, cb+tg*2)    3:(gid+8, cb+tg*2+1)
//   4:(gid, cb+8+tg*2)    5:(gid, cb+8+tg*2+1)
//   6:(gid+8, cb+8+tg*2)  7:(gid+8, cb+8+tg*2+1)
// and the caller reads it back with that same mapping. SWIZZLE states that the
// caller staged sV through swizzled_offset; sP is built by this kernel's own
// scalar stores and is never swizzled. NWARPS must be a compile-time constant so
// the slab loop unrolls and `out` stays in registers.
constexpr int kPvSlabAccums = 8;

// Dim slabs one warp of an NWARPS-warp CTA owns in tile_pv_registers.
template <int HEAD_DIM, int NWARPS> __device__ constexpr int pv_slabs_per_warp() {
    return (HEAD_DIM / 16 + NWARPS - 1) / NWARPS;
}

template <typename scalar_t, int KEY_TILE, int HEAD_DIM, bool BF16, int NWARPS, bool SWIZZLE = false>
__device__ __forceinline__ void
tile_pv_registers(const scalar_t* sP, const scalar_t* sV, const float* row_rescale, float* out, int warp) {
    constexpr int KC = KEY_TILE / 16;
    constexpr int DSLAB = HEAD_DIM / 16;

    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int quad = lane >> 3, rin = lane & 7;
    const int arow = (quad & 1) * 8 + rin, acol = (quad >> 1) * 8;

    uint32_t pa[KC][4];
#pragma unroll
    for (int kc = 0; kc < KC; ++kc) {
        mma::ldmatrix_x4(pa[kc], &sP[arow * KEY_TILE + kc * 16 + acol]);
    }

    const float rescale_low = row_rescale[gid], rescale_high = row_rescale[gid + 8];

#pragma unroll
    for (int slab = 0; slab < pv_slabs_per_warp<HEAD_DIM, NWARPS>(); ++slab) {
        const int ds = warp + slab * NWARPS;
        if (ds >= DSLAB) {
            break;
        }
        float of0[4] = {0, 0, 0, 0}, of1[4] = {0, 0, 0, 0};
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            uint32_t vb[4];
            mma::ldmatrix_x4_trans(vb, &sV[tile_offset<HEAD_DIM, SWIZZLE>(kc * 16 + arow, ds * 16 + acol)]);
            uint32_t b0[2] = {vb[0], vb[1]};  // dims ds*16+0..7
            uint32_t b1[2] = {vb[2], vb[3]};  // dims ds*16+8..15
            mma::mma_m16n8k16<BF16>(of0, pa[kc], b0, of0);
            mma::mma_m16n8k16<BF16>(of1, pa[kc], b1, of1);
        }
        float* acc = out + slab * kPvSlabAccums;
        acc[0] = __fadd_rn(__fmul_rn(acc[0], rescale_low), of0[0]);
        acc[1] = __fadd_rn(__fmul_rn(acc[1], rescale_low), of0[1]);
        acc[2] = __fadd_rn(__fmul_rn(acc[2], rescale_high), of0[2]);
        acc[3] = __fadd_rn(__fmul_rn(acc[3], rescale_high), of0[3]);
        acc[4] = __fadd_rn(__fmul_rn(acc[4], rescale_low), of1[0]);
        acc[5] = __fadd_rn(__fmul_rn(acc[5], rescale_low), of1[1]);
        acc[6] = __fadd_rn(__fmul_rn(acc[6], rescale_high), of1[2]);
        acc[7] = __fadd_rn(__fmul_rn(acc[7], rescale_high), of1[3]);
    }
}

}  // namespace attn
}  // namespace pulsar
