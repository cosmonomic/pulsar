#include "attn_common.cuh"

#include "pulsar/kernels/attn_tiles.cuh"
#include "pulsar/kernels/attn_tuning.hpp"
#include "pulsar/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

// Paged-KV decode attention. KV for one layer lives in a shared pool of
// fixed-size pages; each sequence owns a page table mapping its logical pos to
// physical pages, so many ragged-length sequences share one pool. The slot
// arithmetic is the pool's; see pulsar/runtime/kv/paged_pool.hpp for it.
//
// One layer's slice of the pool, all contiguous:
//   k_pool, v_pool : [num_pages, page_size, n_kv_heads, head_dim]
//   mass_pool      : [num_pages, page_size, n_q_heads]  (fp32, added to)
//   page_tables    : int32 [num_seqs, max_pages]
//   context_lens   : int32 [num_seqs]
//   slot_mapping   : int32 [num_new_tokens]
//   attention_mass_decay : fp32 [num_seqs]  (per-sequence EMA gain alpha)
//   mass_length_gain : length each key's mass is stated against; <= 0 uses each
//                      sequence's own context length
//
// page_size is a compile-time template parameter; the host launcher dispatches on
// the runtime page_size (= k_pool.size(1)).

namespace pulsar {
namespace {

using attn::kMaxHeadDim;
using attn::kThreads;

// Scatter new K/V rows into the pool. Token t (slot = slot_mapping[t]) writes
// into k_pool[slot/PAGE_SIZE, slot%PAGE_SIZE, :, :]. One block per token; the
// n_kv_heads*head_dim elements are strided across the block's threads.
template <typename scalar_t, int PAGE_SIZE>
__global__ void write_kv_kernel(
    scalar_t* __restrict__ k_pool,
    scalar_t* __restrict__ v_pool,
    const scalar_t* __restrict__ k_new,
    const scalar_t* __restrict__ v_new,
    const int32_t* __restrict__ slot_mapping,
    int n_kv_heads,
    int head_dim
) {
    const int t = blockIdx.x;
    const int slot = slot_mapping[t];
    const int page = slot / PAGE_SIZE;
    const int offset = slot % PAGE_SIZE;

    const int64_t n = static_cast<int64_t>(n_kv_heads) * head_dim;
    const int64_t dst = ((static_cast<int64_t>(page) * PAGE_SIZE + offset) * n_kv_heads) * head_dim;
    const int64_t src = static_cast<int64_t>(t) * n;

    for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
        k_pool[dst + e] = k_new[src + e];
        v_pool[dst + e] = v_new[src + e];
    }
}

// Paged decode attention with per-key mass. One block per (seq, query head).
// seq_q == 1: each sequence contributes one new-token query that attends over
// its whole context (no causal masking). Threads stride over the context keys,
// each running an independent numerically-stable online softmax over its own
// subset; the per-thread states are combined with the standard flash rescale.
// A second streaming pass recomputes each normalized weight, scales it by this
// sequence's EMA gain alpha (attention_mass_decay[s]) and by the keys this query
// attended (or by mass_length_gain in their place), and atomicAdds it into mass_pool
// at THIS query head's own column (per query head; no group sum).
// The full score vector is never materialized; everything accumulates in fp32.
template <typename scalar_t, int PAGE_SIZE>
__global__ void attn_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    float* __restrict__ mass_pool,  // in-place +=
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ context_lens,
    const float* __restrict__ attention_mass_decay,  // [num_seqs] EMA gain alpha
    float mass_length_gain,  // <= 0 => the sequence's own context length
    scalar_t* __restrict__ o,
    float* __restrict__ lse_out,  // [num_seqs, n_q_heads]; null when not captured
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int head_dim,
    int group,
    float scale
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int s = blk / n_q_heads;
    const int h = blk % n_q_heads;
    const int g = h / group;

    const int ctx_len = context_lens[s];
    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = q + (static_cast<int64_t>(s) * n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads) {
        qsh[d] = static_cast<float>(qrow[d]);
    }
    __syncthreads();

    // Per-thread online-softmax state over this thread's subset of keys.
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    for (int j = tid; j < ctx_len; j += kThreads) {
        const int phys = seq_table[j / PAGE_SIZE];
        const int offset = j % PAGE_SIZE;
        const int64_t base = ((static_cast<int64_t>(phys) * PAGE_SIZE + offset) * n_kv_heads + g) * head_dim;
        const scalar_t* krow = k_pool + base;
        float sc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sc += qsh[d] * static_cast<float>(krow[d]);
        }
        sc *= scale;

        const float new_m = fmaxf(m, sc);
        const float corr = __expf(m - new_m);  // 0 when m == -inf
        const float p = __expf(sc - new_m);
        l = l * corr + p;
        const scalar_t* vrow = v_pool + base;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + p * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    // Combine per-thread states with the flash rescale: weight each thread's
    // contribution by exp(m_t - M).
    const float M = attn::block_reduce_max<kThreads>(red, tid, m);
    const float w = __expf(m - M);  // 0 when this thread saw no keys
    for (int d = tid; d < head_dim; d += kThreads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = attn::block_reduce_sum<kThreads>(red, tid, l * w);

    if (lse_out && tid == 0) {
        lse_out[static_cast<int64_t>(s) * n_q_heads + h] = M + logf(denom);
    }
    scalar_t* orow = o + (static_cast<int64_t>(s) * n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    // Second streaming pass: recompute each normalized weight and atomicAdd it
    // into mass_pool at the key's physical slot, THIS query head's own column h
    // (per query head; distinct h write distinct slots, no group sum).
    const float mass_gain = attention_mass_decay[s] *
        (mass_length_gain > 0.0f ? mass_length_gain : static_cast<float>(ctx_len));
    for (int j = tid; j < ctx_len; j += kThreads) {
        const int phys = seq_table[j / PAGE_SIZE];
        const int offset = j % PAGE_SIZE;
        const int64_t base = ((static_cast<int64_t>(phys) * PAGE_SIZE + offset) * n_kv_heads + g) * head_dim;
        const scalar_t* krow = k_pool + base;
        float sc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sc += qsh[d] * static_cast<float>(krow[d]);
        }
        sc *= scale;
        const float p = __expf(sc - M) * inv_denom;
        const int64_t mass_idx = (static_cast<int64_t>(phys) * PAGE_SIZE + offset) * n_q_heads + h;
        atomicAdd(&mass_pool[mass_idx], mass_gain * p);
    }
}

// Tensor-core paged decode attention (sm_120 / consumer Blackwell).
//
// Same result as the scalar kernel above, but the two GEMMs (QKᵀ and PV) run on
// the m16n8k16 bf16/fp16 tensor-core MMA (fp32 accumulate). One warp (32 lanes)
// per (sequence s, kv head g); the `group` query heads sharing g occupy MMA rows
// 0..group-1 (rows group..15 are zero-padded and discarded). Context is streamed
// one physical page (PAGE_SIZE keys) at a time; keys past context_lens are
// masked. Online softmax runs per query row in shared memory between the two
// MMAs. A second streaming pass recomputes exp(scale*q·k - LSE) and atomicAdds
// the per-key mass into EACH query head's own mass_pool column (per query head,
// no group sum). The QKᵀ/PV tiles and their fragment layout invariants live in
// attn_tiles.cuh.

// Online-softmax attention over the logical tile range [t_start, t_end) for one
// (seq s, kv head g). Shared by the single-CTA kernel (t range = all tiles) and
// the split kernel (t range = one context slice). Accumulates the RAW,
// UNNORMALIZED state in shared memory: sO (sum of exp(scale*q.k - Mrow)*V, per
// query row), Mrow (running row max), Lrow (running denom). Each scaled score is
// stashed to seq_scores[(logical_key)*group + m] for m < group, for the mass
// pass. The caller owns normalization / LSE. Tiles with no valid keys end the
// loop, so an out-of-range slice leaves sO=0, Mrow=-inf, Lrow=0.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__device__ __forceinline__ void tc_attend_tiles(
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int32_t* __restrict__ seq_table,
    float* __restrict__ seq_scores,
    int ctx_len,
    int g,
    int n_kv_heads,
    int group,
    float scale,
    int t_start,
    int t_end,
    scalar_t* sQ,
    scalar_t* sK,
    scalar_t* sV,
    scalar_t* sP,
    float* sS,
    float* sO,
    float* Mrow,
    float* Lrow
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int64_t stride_slot = static_cast<int64_t>(n_kv_heads) * HEAD_DIM;

    for (int t = t_start; t < t_end; ++t) {
        const int tile_base = t * PAGE_SIZE;
        const int valid = min(PAGE_SIZE, ctx_len - tile_base);
        if (valid <= 0) {
            break;  // no keys left in this slice / sequence
        }
        const int phys = seq_table[t];
        const int64_t slot0 = (static_cast<int64_t>(phys) * PAGE_SIZE) * n_kv_heads + g;
        // Row (page,off,g) base = slot0*HEAD_DIM + off*stride_slot.
        for (int i = tid; i < PAGE_SIZE * HEAD_DIM; i += NWARPS * 32) {
            const int off = i / HEAD_DIM, d = i % HEAD_DIM;
            if (off < valid) {
                const int64_t idx = slot0 * HEAD_DIM + static_cast<int64_t>(off) * stride_slot + d;
                sK[i] = k_pool[idx];
                sV[i] = v_pool[idx];
            } else {
                sK[i] = static_cast<scalar_t>(0);
                sV[i] = static_cast<scalar_t>(0);
            }
        }
        __syncthreads();

        attn::tile_qkt<scalar_t, PAGE_SIZE, HEAD_DIM, BF16>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();

        // Online-softmax update: warp 0's first 16 lanes own the query rows.
        if (warp == 0 && lane < 16) {
            const int m = lane;
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < PAGE_SIZE; ++c) {
                if (c < valid) {
                    tmax = fmaxf(tmax, scale * sS[m * PAGE_SIZE + c]);
                }
            }
            const float newM = fmaxf(Mrow[m], tmax);
            const float corr = __expf(Mrow[m] - newM);  // 0 when Mrow == -inf
            float lsum = Lrow[m] * corr;
#pragma unroll
            for (int c = 0; c < PAGE_SIZE; ++c) {
                float p = 0.0f;
                if (c < valid) {
                    const float ssc = scale * sS[m * PAGE_SIZE + c];
                    p = __expf(ssc - newM);
                    // Stash the scaled score for the mass pass (group rows only).
                    if (m < group) {
                        seq_scores[static_cast<int64_t>(tile_base + c) * group + m] = ssc;
                    }
                }
                sP[m * PAGE_SIZE + c] = static_cast<scalar_t>(p);
                lsum += p;
            }
#pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                sO[m * HEAD_DIM + d] *= corr;
            }
            Mrow[m] = newM;
            Lrow[m] = lsum;
        }
        __syncthreads();

        attn::tile_pv<scalar_t, PAGE_SIZE, HEAD_DIM, BF16>(sP, sV, sO, warp, NWARPS);
        __syncthreads();
    }
}

template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) attn_decode_tc_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    float* __restrict__ mass_pool,
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ context_lens,
    const float* __restrict__ attention_mass_decay,  // [num_seqs] EMA gain alpha
    float mass_length_gain,  // <= 0 => the sequence's own context length
    scalar_t* __restrict__ o,
    float* __restrict__ lse_global,  // [num_seqs, n_q_heads]; null when not captured
    float* __restrict__ scores,  // scratch: pass-1 scaled QKᵀ, read in pass 2
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    float scale
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int blk = blockIdx.x;
    const int s = blk / n_kv_heads;
    const int g = blk % n_kv_heads;

    const int ctx_len = context_lens[s];
    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;
    const int ntiles = (ctx_len + PAGE_SIZE - 1) / PAGE_SIZE;

    // Scratch base for this (seq, kv head); inner layout [logical_key][query_row].
    float* seq_scores = scores + static_cast<int64_t>(s) * scores_stride_seq + g * scores_stride_head;

    __shared__ scalar_t sQ[16 * HEAD_DIM];
    __shared__ scalar_t sK[PAGE_SIZE * HEAD_DIM];
    __shared__ scalar_t sV[PAGE_SIZE * HEAD_DIM];
    __shared__ scalar_t sP[16 * PAGE_SIZE];
    __shared__ float sS[16 * PAGE_SIZE];
    __shared__ float sO[16 * HEAD_DIM];
    __shared__ float Mrow[16], Lrow[16], LSE[16];

    // Load Q for the group's heads into MMA rows; pad rows >= group with zero.
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < group) {
            const int h = g * group + row;
            v = q[(static_cast<int64_t>(s) * n_q_heads + h) * HEAD_DIM + col];
        }
        sQ[i] = v;
    }
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        sO[i] = 0.0f;
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
    }
    __syncthreads();

    // Pass 1: online-softmax attention over ALL tiles -> sO, Mrow, Lrow.
    tc_attend_tiles<scalar_t, PAGE_SIZE, HEAD_DIM, BF16, NWARPS>(
        k_pool,
        v_pool,
        seq_table,
        seq_scores,
        ctx_len,
        g,
        n_kv_heads,
        group,
        scale,
        0,
        ntiles,
        sQ,
        sK,
        sV,
        sP,
        sS,
        sO,
        Mrow,
        Lrow
    );

    // Normalize and write o; stash LSE for the mass pass.
    if (tid < 16 && tid < group) {
        const int m = tid;
        const float invl = 1.0f / Lrow[m];
        const int h = g * group + m;
        scalar_t* orow = o + (static_cast<int64_t>(s) * n_q_heads + h) * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            orow[d] = static_cast<scalar_t>(sO[m * HEAD_DIM + d] * invl);
        }
        LSE[m] = Mrow[m] + logf(Lrow[m]);
        if (lse_global) {
            lse_global[static_cast<int64_t>(s) * n_q_heads + h] = LSE[m];
        }
    }
    __syncthreads();

    // Pass 2: read the stashed scores, accumulate per-key mass into EACH query
    // head's own column (per query head, no group sum). No K reload / QK^T recompute:
    // pass 1 already stored scale*q.k.
    const float mass_gain = attention_mass_decay[s] *
        (mass_length_gain > 0.0f ? mass_length_gain : static_cast<float>(ctx_len));
    for (int t = 0; t < ntiles; ++t) {
        const int tile_base = t * PAGE_SIZE;
        const int valid = min(PAGE_SIZE, ctx_len - tile_base);
        const int phys = seq_table[t];

        // One thread per valid key column writes the group's query heads separately.
        if (warp == 0 && lane < valid) {
            const int c = lane;
            const float* col = seq_scores + static_cast<int64_t>(tile_base + c) * group;
            for (int m = 0; m < group; ++m) {
                const int h = g * group + m;
                const float p = __expf(col[m] - LSE[m]);
                const int64_t mass_idx = (static_cast<int64_t>(phys) * PAGE_SIZE + c) * n_q_heads + h;
                atomicAdd(&mass_pool[mass_idx], mass_gain * p);
            }
        }
    }
}

// Split-K (flash-decode) tensor-core path: splits each (seq, kv head)'s context
// over `nsplits` CTAs across three kernels.
//   (a) split kernel : grid = num_seqs*n_kv_heads*nsplits. Each CTA runs the
//       online softmax over one context slice and writes the RAW per-slice
//       state (o_partial, m_partial, l_partial) plus the slice's stashed scores.
//   (b) combine kernel: grid = num_seqs*n_kv_heads. Reduces the nsplits partials
//       per query row with the flash rescale -> normalized o + LSE.
//   (c) mass kernel   : grid = num_seqs*n_kv_heads. exp(score - LSE) per key,
//       atomicAdd into EACH query head's own mass_pool column (per query head).
// Partial buffer layouts (fp32), all for one (seq s, kv head g, split i, row m):
//   o_partial[s][g][i][m][d] : [num_seqs, n_kv_heads, nsplits, group, head_dim]
//   m_partial[s][g][i][m]    : [num_seqs, n_kv_heads, nsplits, group]
//   l_partial[s][g][i][m]    : [num_seqs, n_kv_heads, nsplits, group]
//   lse_out[s][g][m]         : [num_seqs, n_kv_heads, group]
// scores scratch is the same [num_seqs, n_kv_heads, kv_capacity, group] buffer as
// the single-CTA path; each split owns disjoint keys, so no contention.

template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) paged_decode_split_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    float* __restrict__ o_partial,
    float* __restrict__ m_partial,
    float* __restrict__ l_partial,
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    float scale,
    int nsplits,
    int tps
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int split = blk % nsplits;
    const int sg = blk / nsplits;
    const int g = sg % n_kv_heads;
    const int s = sg / n_kv_heads;

    const int ctx_len = context_lens[s];
    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;
    float* seq_scores = scores + static_cast<int64_t>(s) * scores_stride_seq + g * scores_stride_head;

    __shared__ scalar_t sQ[16 * HEAD_DIM];
    __shared__ scalar_t sK[PAGE_SIZE * HEAD_DIM];
    __shared__ scalar_t sV[PAGE_SIZE * HEAD_DIM];
    __shared__ scalar_t sP[16 * PAGE_SIZE];
    __shared__ float sS[16 * PAGE_SIZE];
    __shared__ float sO[16 * HEAD_DIM];
    __shared__ float Mrow[16], Lrow[16];

    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < group) {
            const int h = g * group + row;
            v = q[(static_cast<int64_t>(s) * n_q_heads + h) * HEAD_DIM + col];
        }
        sQ[i] = v;
    }
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        sO[i] = 0.0f;
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
    }
    __syncthreads();

    const int t_start = split * tps;
    const int t_end = t_start + tps;
    tc_attend_tiles<scalar_t, PAGE_SIZE, HEAD_DIM, BF16, NWARPS>(
        k_pool,
        v_pool,
        seq_table,
        seq_scores,
        ctx_len,
        g,
        n_kv_heads,
        group,
        scale,
        t_start,
        t_end,
        sQ,
        sK,
        sV,
        sP,
        sS,
        sO,
        Mrow,
        Lrow
    );
    __syncthreads();

    // Write the raw per-slice state. An empty slice keeps Mrow=-inf, Lrow=0,
    // sO=0, which the combine kernel ignores.
    const int64_t base = (static_cast<int64_t>(sg) * nsplits + split) * group;
    if (tid < group) {
        m_partial[base + tid] = Mrow[tid];
        l_partial[base + tid] = Lrow[tid];
    }
    for (int i = tid; i < group * HEAD_DIM; i += NWARPS * 32) {
        const int m = i / HEAD_DIM, d = i % HEAD_DIM;
        o_partial[base * HEAD_DIM + i] = sO[m * HEAD_DIM + d];
    }
}

// Combine the nsplits partials per query row (flash rescale) -> normalized o and
// LSE. One CTA per (seq, kv head).
template <typename scalar_t>
__global__ void paged_decode_combine_kernel(
    scalar_t* __restrict__ o,
    float* __restrict__ lse_out,
    const float* __restrict__ o_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    int n_q_heads,
    int n_kv_heads,
    int nsplits,
    int group,
    int head_dim
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int g = blk % n_kv_heads;
    const int s = blk / n_kv_heads;
    const int64_t sg = static_cast<int64_t>(s) * n_kv_heads + g;
    const float* mp = m_partial + sg * nsplits * group;
    const float* lp = l_partial + sg * nsplits * group;
    const float* op = o_partial + sg * nsplits * group * head_dim;

    for (int m = 0; m < group; ++m) {
        float M = -CUDART_INF_F;
        for (int i = 0; i < nsplits; ++i) {
            M = fmaxf(M, mp[i * group + m]);
        }
        float l = 0.0f;
        for (int i = 0; i < nsplits; ++i) {
            const float mi = mp[i * group + m];
            if (mi > -CUDART_INF_F) {
                l += lp[i * group + m] * __expf(mi - M);
            }
        }
        const bool bad = (l == 0.0f) || !isfinite(M);
        const float invl = bad ? 0.0f : 1.0f / l;

        const int h = g * group + m;
        scalar_t* orow = o + (static_cast<int64_t>(s) * n_q_heads + h) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            if (!bad) {
                for (int i = 0; i < nsplits; ++i) {
                    const float mi = mp[i * group + m];
                    if (mi > -CUDART_INF_F) {
                        acc += op[(static_cast<int64_t>(i) * group + m) * head_dim + d] * __expf(mi - M);
                    }
                }
            }
            orow[d] = static_cast<scalar_t>(acc * invl);
        }
        if (tid == 0) {
            lse_out[sg * group + m] = bad ? -CUDART_INF_F : (M + logf(l));
        }
    }
}

// Per-key mass from the stashed scores and LSE. One CTA per (seq, kv head);
// page_size is a runtime value here (the score layout is independent of it).
__global__ void paged_decode_mass_kernel(
    float* __restrict__ mass_pool,
    const float* __restrict__ scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    const float* __restrict__ lse_out,
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ context_lens,
    const float* __restrict__ attention_mass_decay,  // [num_seqs] EMA gain alpha
    float mass_length_gain,  // <= 0 => the sequence's own context length
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int page_size,
    int group
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int g = blk % n_kv_heads;
    const int s = blk / n_kv_heads;
    const int ctx_len = context_lens[s];
    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;
    const float* seq_scores = scores + static_cast<int64_t>(s) * scores_stride_seq + g * scores_stride_head;
    const int64_t sg = static_cast<int64_t>(s) * n_kv_heads + g;
    const float* lse = lse_out + sg * group;

    // Each key's normalized weight is written to EACH query head's own mass column
    // (per query head, no group sum).
    const float mass_gain = attention_mass_decay[s] *
        (mass_length_gain > 0.0f ? mass_length_gain : static_cast<float>(ctx_len));
    for (int j = tid; j < ctx_len; j += blockDim.x) {
        const int phys = seq_table[j / page_size];
        const int off = j % page_size;
        const float* col = seq_scores + static_cast<int64_t>(j) * group;
        for (int m = 0; m < group; ++m) {
            const int h = g * group + m;
            const float p = __expf(col[m] - lse[m]);
            const int64_t mass_idx = (static_cast<int64_t>(phys) * page_size + off) * n_q_heads + h;
            atomicAdd(&mass_pool[mass_idx], mass_gain * p);
        }
    }
}

}  // namespace

void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& slot_mapping
) {
    TORCH_CHECK(
        k_pool.is_cuda() && v_pool.is_cuda() && k_new.is_cuda() && v_new.is_cuda() && slot_mapping.is_cuda(),
        "write_kv: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        k_pool.scalar_type() == v_pool.scalar_type() && k_pool.scalar_type() == k_new.scalar_type() &&
            k_new.scalar_type() == v_new.scalar_type(),
        "write_kv: k_pool, v_pool, k_new, v_new must share a dtype"
    );
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "write_kv: slot_mapping must be int32");
    TORCH_CHECK(
        k_pool.dim() == 4 && v_pool.dim() == 4,
        "write_kv: pools must be 4-D [num_pages, page_size, n_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        k_new.dim() == 3 && v_new.dim() == 3,
        "write_kv: k_new, v_new must be 3-D [num_new_tokens, n_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        k_pool.is_contiguous() && v_pool.is_contiguous(),
        "write_kv: pools must be contiguous (mutated in place)"
    );

    const int64_t page_size = k_pool.size(1);
    const int64_t n_kv_heads = k_pool.size(2);
    const int64_t head_dim = k_pool.size(3);
    const int64_t num_new_tokens = k_new.size(0);

    TORCH_CHECK(
        v_pool.size(1) == page_size && v_pool.size(2) == n_kv_heads && v_pool.size(3) == head_dim,
        "write_kv: v_pool must match k_pool shape"
    );
    TORCH_CHECK(
        k_new.size(1) == n_kv_heads && k_new.size(2) == head_dim,
        "write_kv: k_new heads/head_dim must match the pool"
    );
    TORCH_CHECK(
        v_new.size(0) == num_new_tokens && v_new.size(1) == n_kv_heads && v_new.size(2) == head_dim,
        "write_kv: v_new must match k_new shape"
    );
    TORCH_CHECK(
        slot_mapping.dim() == 1 && slot_mapping.size(0) == num_new_tokens,
        "write_kv: slot_mapping must be 1-D [num_new_tokens]"
    );

    if (num_new_tokens == 0) {
        return;
    }

    auto kn = k_new.contiguous();
    auto vn = v_new.contiguous();
    auto sm = slot_mapping.contiguous();

    const int threads = static_cast<int>(std::min<int64_t>(256, n_kv_heads * head_dim));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, k_pool.scalar_type(), "write_kv_cuda", [&] {
        attn::dispatch_page_size("write_kv", page_size, [&](auto page_size_tag) {
            constexpr int BS = decltype(page_size_tag)::value;
            write_kv_kernel<scalar_t, BS><<<static_cast<int>(num_new_tokens), threads, 0, stream>>>(
                k_pool.data_ptr<scalar_t>(),
                v_pool.data_ptr<scalar_t>(),
                kn.data_ptr<scalar_t>(),
                vn.data_ptr<scalar_t>(),
                sm.data_ptr<int32_t>(),
                static_cast<int>(n_kv_heads),
                static_cast<int>(head_dim)
            );
        });
    });
}

namespace {

// Instantiate + launch the single-CTA tensor-core kernel for the runtime
// (page_size, head_dim). scalar_t/BF16 are fixed by the caller's dtype branch.
template <typename scalar_t, bool BF16>
void launch_tc_decode(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    float* mass,
    const int32_t* bt,
    const int32_t* cl,
    const float* decay,
    float mass_length_gain,
    scalar_t* o,
    float* lse_global,
    float* scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    int head_dim,
    int page_size,
    float scale,
    int64_t tc_blocks,
    cudaStream_t stream
) {
    attn::dispatch_page_head("attn_decode", page_size, head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            attn_decode_tc_kernel<scalar_t, BS, HD, BF16, NWARPS>
                <<<static_cast<int>(tc_blocks), NWARPS * 32, 0, stream>>>(
                    q,
                    k,
                    v,
                    mass,
                    bt,
                    cl,
                    decay,
                    mass_length_gain,
                    o,
                    lse_global,
                    scores,
                    scores_stride_seq,
                    scores_stride_head,
                    n_q_heads,
                    n_kv_heads,
                    max_pages,
                    group,
                    scale
                );
        });
    });
}

// Instantiate + launch the split (flash-decode) kernel for the runtime
// (page_size, head_dim). Grid = split_blocks = num_seqs*n_kv_heads*nsplits.
template <typename scalar_t, bool BF16>
void launch_tc_split(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    const int32_t* bt,
    const int32_t* cl,
    float* scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    float* o_partial,
    float* m_partial,
    float* l_partial,
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    int head_dim,
    int page_size,
    float scale,
    int nsplits,
    int tps,
    int64_t split_blocks,
    cudaStream_t stream
) {
    attn::dispatch_page_head("attn_decode_split", page_size, head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            paged_decode_split_kernel<scalar_t, BS, HD, BF16, NWARPS>
                <<<static_cast<int>(split_blocks), NWARPS * 32, 0, stream>>>(
                    q,
                    k,
                    v,
                    bt,
                    cl,
                    scores,
                    scores_stride_seq,
                    scores_stride_head,
                    o_partial,
                    m_partial,
                    l_partial,
                    n_q_heads,
                    n_kv_heads,
                    max_pages,
                    group,
                    scale,
                    nsplits,
                    tps
                );
        });
    });
}

// Shared body of the tensor-core op and its scalar-only reference sibling.
// force_scalar routes everything through the scalar kernel; the default op uses
// the tensor-core kernel for fp16/bf16 with head_dim in {64,128} and page_size in
// {16,32}, and falls back to the scalar kernel otherwise. forced_splits > 0 forces
// the split path with that many splits (1 included); forced_splits == 0 picks
// nsplits from an occupancy heuristic and uses the single-CTA kernel when it comes
// out 1. forced_splits is only consulted on the tensor-core path.
at::Tensor paged_decode_impl(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain,
    bool force_scalar,
    const at::Tensor& lse_capture = {},
    int forced_splits = 0
) {
    TORCH_CHECK(
        q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && mass_pool.is_cuda() && page_tables.is_cuda() &&
            context_lens.is_cuda() && attention_mass_decay.is_cuda(),
        "attn_decode: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        attention_mass_decay.scalar_type() == at::kFloat && attention_mass_decay.dim() == 1,
        "attn_decode: attention_mass_decay must be fp32 1-D [num_seqs]"
    );
    // Optional per-(query row, query head) logsumexp capture, fp32 and contiguous.
    TORCH_CHECK(
        !lse_capture.defined() ||
            (lse_capture.is_cuda() && lse_capture.scalar_type() == at::kFloat && lse_capture.is_contiguous()),
        "attn_decode: lse capture must be a contiguous fp32 CUDA tensor"
    );
    float* lse_ptr = lse_capture.defined() ? lse_capture.data_ptr<float>() : nullptr;
    TORCH_CHECK(
        q.scalar_type() == k_pool.scalar_type() && k_pool.scalar_type() == v_pool.scalar_type(),
        "attn_decode: q, k_pool, v_pool must share a dtype"
    );
    TORCH_CHECK(mass_pool.scalar_type() == at::kFloat, "attn_decode: mass_pool must be fp32");
    TORCH_CHECK(
        page_tables.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
        "attn_decode: page_tables and context_lens must be int32"
    );
    TORCH_CHECK(q.dim() == 3, "attn_decode: q must be 3-D [num_seqs, n_q_heads, head_dim]");
    TORCH_CHECK(
        k_pool.dim() == 4 && v_pool.dim() == 4,
        "attn_decode: pools must be 4-D [num_pages, page_size, n_kv_heads, head_dim]"
    );
    TORCH_CHECK(mass_pool.dim() == 3, "attn_decode: mass_pool must be 3-D [num_pages, page_size, n_q_heads]");
    TORCH_CHECK(page_tables.dim() == 2, "attn_decode: page_tables must be 2-D [num_seqs, max_pages]");
    TORCH_CHECK(context_lens.dim() == 1, "attn_decode: context_lens must be 1-D [num_seqs]");

    const int64_t num_seqs = q.size(0);
    const int64_t n_q_heads = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t num_pages = k_pool.size(0);
    const int64_t page_size = k_pool.size(1);
    const int64_t n_kv_heads = k_pool.size(2);
    const int64_t max_pages = page_tables.size(1);

    TORCH_CHECK(
        v_pool.size(0) == num_pages && v_pool.size(1) == page_size && v_pool.size(2) == n_kv_heads &&
            v_pool.size(3) == head_dim,
        "attn_decode: v_pool must match k_pool shape"
    );
    TORCH_CHECK(
        mass_pool.size(0) == num_pages && mass_pool.size(1) == page_size && mass_pool.size(2) == n_q_heads,
        "attn_decode: mass_pool must be [num_pages, page_size, n_q_heads]"
    );
    TORCH_CHECK(k_pool.size(3) == head_dim, "attn_decode: pool head_dim must match q");
    TORCH_CHECK(
        page_tables.size(0) == num_seqs && context_lens.size(0) == num_seqs,
        "attn_decode: page_tables and context_lens must have num_seqs rows"
    );
    TORCH_CHECK(
        attention_mass_decay.size(0) == num_seqs,
        "attn_decode: attention_mass_decay has ",
        attention_mass_decay.size(0),
        " entries for ",
        num_seqs,
        " sequences"
    );
    TORCH_CHECK(
        n_kv_heads > 0 && n_q_heads % n_kv_heads == 0,
        "attn_decode: n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(head_dim <= kMaxHeadDim, "attn_decode: head_dim exceeds compile-time max");
    TORCH_CHECK(
        k_pool.is_contiguous() && v_pool.is_contiguous() && mass_pool.is_contiguous(),
        "attn_decode: pools must be contiguous"
    );

    auto qc = q.contiguous();
    auto bt = page_tables.contiguous();
    auto cl = context_lens.contiguous();
    auto decay = attention_mass_decay.contiguous();
    auto o = at::empty_like(qc);

    if (num_seqs == 0) {
        return o;
    }

    const float gain = static_cast<float>(mass_length_gain);
    const int group = static_cast<int>(n_q_heads / n_kv_heads);
    // EMA gain: the received per-key mass is scaled by the owning sequence's alpha
    // before accumulation. alpha IS the EMA rate (== the decay the upkeep retains
    // 1-alpha of); alpha == 1 recovers the raw weight.
    const int64_t blocks = num_seqs * n_q_heads;
    const size_t smem = static_cast<size_t>(2 * head_dim + kThreads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto dt = qc.scalar_type();
    const bool tc_ok = !force_scalar && (dt == at::kHalf || dt == at::kBFloat16) &&
        (head_dim == 64 || head_dim == 128) && (page_size == 16 || page_size == 32);

    if (tc_ok) {
        // Scratch for pass-1 scaled scores, keyed by (seq, kv head, logical key,
        // query row); sized by batch x per-seq KV capacity, not pool size.
        const int64_t kv_capacity = max_pages * page_size;
        auto fopts = qc.options().dtype(at::kFloat);
        auto scores = at::empty({num_seqs, n_kv_heads, kv_capacity, group}, fopts);
        const int64_t scores_stride_head = kv_capacity * group;
        const int64_t scores_stride_seq = n_kv_heads * scores_stride_head;

        // Split count targeting ~2 CTAs per SM. A split must own at least two
        // tiles, so nsplits is capped at ceil(max_pages/2), and at 32 overall.
        int nsplits;
        bool use_split;
        if (forced_splits > 0) {
            nsplits = forced_splits;
            use_split = true;
        } else {
            const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
            const int64_t base_ctas = num_seqs * n_kv_heads;
            int64_t ns = (2LL * sm_count + base_ctas - 1) / base_ctas;
            const int64_t cap_by_tiles = (max_pages + 1) / 2;  // ceil(cap/2)
            ns = std::min(ns, std::max<int64_t>(cap_by_tiles, 1));
            ns = std::min<int64_t>(ns, 32);
            ns = std::max<int64_t>(ns, 1);
            nsplits = static_cast<int>(ns);
            use_split = nsplits > 1;
        }

        if (!use_split) {
            // Single-CTA path: one CTA per (seq, kv head).
            const int64_t tc_blocks = num_seqs * n_kv_heads;
            if (dt == at::kBFloat16) {
                launch_tc_decode<at::BFloat16, true>(
                    qc.data_ptr<at::BFloat16>(),
                    k_pool.data_ptr<at::BFloat16>(),
                    v_pool.data_ptr<at::BFloat16>(),
                    mass_pool.data_ptr<float>(),
                    bt.data_ptr<int32_t>(),
                    cl.data_ptr<int32_t>(),
                    decay.data_ptr<float>(),
                    gain,
                    o.data_ptr<at::BFloat16>(),
                    lse_ptr,
                    scores.data_ptr<float>(),
                    scores_stride_seq,
                    scores_stride_head,
                    static_cast<int>(n_q_heads),
                    static_cast<int>(n_kv_heads),
                    static_cast<int>(max_pages),
                    group,
                    static_cast<int>(head_dim),
                    static_cast<int>(page_size),
                    static_cast<float>(scale),
                    tc_blocks,
                    stream
                );
            } else {
                launch_tc_decode<at::Half, false>(
                    qc.data_ptr<at::Half>(),
                    k_pool.data_ptr<at::Half>(),
                    v_pool.data_ptr<at::Half>(),
                    mass_pool.data_ptr<float>(),
                    bt.data_ptr<int32_t>(),
                    cl.data_ptr<int32_t>(),
                    decay.data_ptr<float>(),
                    gain,
                    o.data_ptr<at::Half>(),
                    lse_ptr,
                    scores.data_ptr<float>(),
                    scores_stride_seq,
                    scores_stride_head,
                    static_cast<int>(n_q_heads),
                    static_cast<int>(n_kv_heads),
                    static_cast<int>(max_pages),
                    group,
                    static_cast<int>(head_dim),
                    static_cast<int>(page_size),
                    static_cast<float>(scale),
                    tc_blocks,
                    stream
                );
            }
            return o;
        }

        // Split (flash-decode) path: three kernels.
        const int tps = static_cast<int>((max_pages + nsplits - 1) / nsplits);
        const int64_t split_blocks = num_seqs * n_kv_heads * static_cast<int64_t>(nsplits);
        const int64_t combine_blocks = num_seqs * n_kv_heads;
        auto o_partial = at::empty({num_seqs, n_kv_heads, nsplits, group, head_dim}, fopts);
        auto m_partial = at::empty({num_seqs, n_kv_heads, nsplits, group}, fopts);
        auto l_partial = at::empty({num_seqs, n_kv_heads, nsplits, group}, fopts);
        // The combine kernel's [num_seqs, n_kv_heads, group] layout flattens to
        // [num_seqs, n_q_heads], so a caller-provided buffer is written directly.
        auto lse_out = lse_capture.defined() ? lse_capture.view({num_seqs, n_kv_heads, group})
                                             : at::empty({num_seqs, n_kv_heads, group}, fopts);

        if (dt == at::kBFloat16) {
            launch_tc_split<at::BFloat16, true>(
                qc.data_ptr<at::BFloat16>(),
                k_pool.data_ptr<at::BFloat16>(),
                v_pool.data_ptr<at::BFloat16>(),
                bt.data_ptr<int32_t>(),
                cl.data_ptr<int32_t>(),
                scores.data_ptr<float>(),
                scores_stride_seq,
                scores_stride_head,
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                group,
                static_cast<int>(head_dim),
                static_cast<int>(page_size),
                static_cast<float>(scale),
                nsplits,
                tps,
                split_blocks,
                stream
            );
            paged_decode_combine_kernel<at::BFloat16><<<static_cast<int>(combine_blocks), kThreads, 0, stream>>>(
                o.data_ptr<at::BFloat16>(),
                lse_out.data_ptr<float>(),
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                nsplits,
                group,
                static_cast<int>(head_dim)
            );
        } else {
            launch_tc_split<at::Half, false>(
                qc.data_ptr<at::Half>(),
                k_pool.data_ptr<at::Half>(),
                v_pool.data_ptr<at::Half>(),
                bt.data_ptr<int32_t>(),
                cl.data_ptr<int32_t>(),
                scores.data_ptr<float>(),
                scores_stride_seq,
                scores_stride_head,
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                group,
                static_cast<int>(head_dim),
                static_cast<int>(page_size),
                static_cast<float>(scale),
                nsplits,
                tps,
                split_blocks,
                stream
            );
            paged_decode_combine_kernel<at::Half><<<static_cast<int>(combine_blocks), kThreads, 0, stream>>>(
                o.data_ptr<at::Half>(),
                lse_out.data_ptr<float>(),
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                nsplits,
                group,
                static_cast<int>(head_dim)
            );
        }
        paged_decode_mass_kernel<<<static_cast<int>(combine_blocks), kThreads, 0, stream>>>(
            mass_pool.data_ptr<float>(),
            scores.data_ptr<float>(),
            scores_stride_seq,
            scores_stride_head,
            lse_out.data_ptr<float>(),
            bt.data_ptr<int32_t>(),
            cl.data_ptr<int32_t>(),
            decay.data_ptr<float>(),
            gain,
            static_cast<int>(n_q_heads),
            static_cast<int>(n_kv_heads),
            static_cast<int>(max_pages),
            static_cast<int>(page_size),
            group
        );
        return o;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qc.scalar_type(), "attn_decode_cuda", [&] {
        attn::dispatch_page_size("attn_decode", page_size, [&](auto page_size_tag) {
            constexpr int BS = decltype(page_size_tag)::value;
            attn_decode_kernel<scalar_t, BS><<<static_cast<int>(blocks), kThreads, smem, stream>>>(
                qc.data_ptr<scalar_t>(),
                k_pool.data_ptr<scalar_t>(),
                v_pool.data_ptr<scalar_t>(),
                mass_pool.data_ptr<float>(),
                bt.data_ptr<int32_t>(),
                cl.data_ptr<int32_t>(),
                decay.data_ptr<float>(),
                gain,
                o.data_ptr<scalar_t>(),
                lse_ptr,
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                static_cast<int>(head_dim),
                group,
                static_cast<float>(scale)
            );
        });
    });
    return o;
}

}  // namespace

at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture,
    double mass_length_gain
) {
    return paged_decode_impl(
        q,
        k_pool,
        v_pool,
        mass_pool,
        page_tables,
        context_lens,
        scale,
        attention_mass_decay,
        mass_length_gain,
        /*force_scalar=*/false,
        lse_capture.has_value() ? *lse_capture : at::Tensor{}
    );
}

at::Tensor attn_decode_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain
) {
    return paged_decode_impl(
        q,
        k_pool,
        v_pool,
        mass_pool,
        page_tables,
        context_lens,
        scale,
        attention_mass_decay,
        mass_length_gain,
        /*force_scalar=*/true
    );
}

at::Tensor attn_decode_split_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    int64_t num_splits,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture,
    double mass_length_gain
) {
    TORCH_CHECK(num_splits >= 1, "attn_decode_split: num_splits must be >= 1");
    return paged_decode_impl(
        q,
        k_pool,
        v_pool,
        mass_pool,
        page_tables,
        context_lens,
        scale,
        attention_mass_decay,
        mass_length_gain,
        /*force_scalar=*/false,
        lse_capture.has_value() ? *lse_capture : at::Tensor{},
        static_cast<int>(num_splits)
    );
}

}  // namespace pulsar
