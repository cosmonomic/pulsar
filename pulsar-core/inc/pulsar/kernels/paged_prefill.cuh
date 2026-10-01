#pragma once

#include "pulsar/kernels/attn_tiles.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

// Tensor-core prefill attention. Flash-attention-style: one CTA per (query-tile,
// q head); a query tile is up to 16 query tokens of ONE q head, occupying MMA
// rows 0..tile_rows-1 (rows tile_rows..15 are zero-padded and discarded). The
// causal key range is streamed KEY_PAGES physical pages (KEY_PAGES * PAGE_SIZE
// keys) at a time; the causal mask is applied per (query row, key) in the online
// softmax. A second streaming pass recomputes QKᵀ and atomicAdds the per-key mass
// (summed over this tile's query rows with p >= kp, all for the ONE q head) into
// this q head's own mass_pool column (per query head). The QKᵀ/PV tiles live in
// attn_tiles.cuh; the op wrapper and the argument shapes are in
// kernels/paged_prefill.cu.
//
// NWARPS, KEY_PAGES and CTAS_PER_SM are launch shape, not geometry: every one of
// them is device specific and comes from the caller's tuning row
// (pulsar/kernels/attn_tuning.hpp). KEY_PAGES only changes the order in which the online
// softmax folds keys in, so o, lse and mass move in their last bits with it.
// CTAS_PER_SM is the occupancy floor ptxas must fit the register allocation to; 1
// leaves it unconstrained.

namespace pulsar {
namespace attn {

template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS, int KEY_PAGES, int CTAS_PER_SM>
__global__ void __launch_bounds__(NWARPS * 32, CTAS_PER_SM) attn_prefill_tc_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    float* __restrict__ mass_pool,  // null skips the mass pass
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seqlens_k,
    const float* __restrict__ attention_mass_decay,  // [num_seqs] EMA gain alpha
    float mass_length_gain,  // <= 0 => each row's own causal key count
    const int32_t* __restrict__ tile_seq,  // [ntiles] seq owning each query-tile
    const int32_t* __restrict__ tile_qbase,  // [ntiles] local query start of tile
    scalar_t* __restrict__ o,
    float* __restrict__ lse_global,  // [total_q, n_q_heads]; null when not captured
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    float scale
) {
    constexpr int KEY_TILE = PAGE_SIZE * KEY_PAGES;
    // sS row stride. A stride of exactly KEY_TILE puts every row of a column on one
    // shared bank, and the two warp-0 sweeps below read a whole column at once.
    constexpr int SCORE_STRIDE = KEY_TILE + 1;
    // The per-row max and denominator sweeps give every query row a group of
    // kSweepLanes consecutive threads and reduce the group on shuffles. A group's
    // threads take consecutive score pairs, which keeps the two half-warps of a
    // warp on disjoint banks at SCORE_STRIDE odd.
    constexpr int kSweepLanes = 4;
    constexpr int kSweepCols = KEY_TILE / kSweepLanes;
    constexpr int kSweepThreads = 16 * kSweepLanes;
    static_assert(
        kSweepThreads % 32 == 0 && kSweepThreads <= NWARPS * 32,
        "the sweep groups must fill whole warps of the CTA"
    );
    static_assert(KEY_TILE % kSweepLanes == 0, "key tile must split over a group");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sweep_row = tid / kSweepLanes;
    const int sweep_lane = tid % kSweepLanes;
    const int blk = blockIdx.x;
    const int tile = blk / n_q_heads;
    const int h = blk % n_q_heads;
    const int g = h / group;

    const int s = tile_seq[tile];
    // This sequence's EMA gain alpha and the retention 1 - alpha it implies.
    const float mass_decay = attention_mass_decay[s];
    const float retention = 1.0f - mass_decay;
    const int qbase = tile_qbase[tile];
    const int q0 = cu_seqlens_q[s];
    const int seq_q = cu_seqlens_q[s + 1] - q0;
    const int ctx_len = seqlens_k[s];
    const int ctx_start = ctx_len - seq_q;
    const int tile_rows = min(16, seq_q - qbase);
    const int q_global_base = q0 + qbase;
    // Context pos of MMA row m is ctx_start + qbase + m; the tile's highest.
    const int max_p = ctx_start + qbase + tile_rows - 1;
    const int n_key_tiles = max_p / KEY_TILE + 1;  // tiles covering keys [0, max_p]

    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;
    const int64_t stride_slot = static_cast<int64_t>(n_kv_heads) * HEAD_DIM;

    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    constexpr int kChunkElems = 16 / sizeof(scalar_t);
    constexpr int kRowChunks = HEAD_DIM / kChunkElems;
    constexpr int kPageChunks = PAGE_SIZE * kRowChunks;

    // The ldmatrix row addresses require every tile to be 16-byte aligned. sQ, sK
    // and sV carry their columns permuted by attn::swizzled_offset; sP and sS are
    // reached by scalar accesses only and stay row-major.
    __shared__ __align__(16) scalar_t sQ[16 * HEAD_DIM];
    __shared__ __align__(16) scalar_t sK[KEY_TILE * HEAD_DIM];
    __shared__ __align__(16) scalar_t sV[KEY_TILE * HEAD_DIM];
    __shared__ __align__(16) scalar_t sP[16 * KEY_TILE];
    __shared__ __align__(16) float sS[16 * SCORE_STRIDE];
    __shared__ float Mrow[16], Lrow[16], LSE[16];
    // Online-softmax rescale of the row's running output, published so every warp
    // can fold it into its own share of the accumulator.
    __shared__ float row_rescale[16];
    __shared__ int qpos[16];  // context pos of each MMA row, -1 if padding
    // Per-query multiplier on the mass this row hands out: the within-chunk retention
    // grading retention^(offset of this row from the chunk end), so a key's accumulated
    // mass is the per-token EMA of the attention it received (older queries in the
    // chunk weigh less; retention == 1 -> flat), times the length gain over the
    // qpos[m] + 1 keys THAT row attends, or over mass_length_gain in their place.
    __shared__ float mass_weight[16];

    // Load Q for this tile's rows into the MMA M dimension; pad rows >= tile_rows.
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < tile_rows) {
            const int64_t qi = (static_cast<int64_t>(q_global_base + row) * n_q_heads + h) * HEAD_DIM;
            v = q[qi + col];
        }
        sQ[attn::swizzled_offset<HEAD_DIM>(row, col)] = v;
    }
    // This lane's share of the running output, one attn::kPvSlabAccums block per dim
    // slab this warp owns. attn::tile_pv_registers states the (row, column) each
    // entry carries.
    constexpr int kSlabs = attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>();
    float out_acc[kSlabs * attn::kPvSlabAccums] = {};
    for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
        sP[i] = static_cast<scalar_t>(0);
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
        // Padding rows never enter the softmax update, so their output must stay put.
        row_rescale[tid] = 1.0f;
        const int row_pos = ctx_start + qbase + tid;  // this row's context pos
        qpos[tid] = (tid < tile_rows) ? row_pos : -1;
        // Offset from the chunk end: last query in the chunk (seq_q-1) gets
        // retention^0. The row attends the row_pos + 1 causal keys [0, row_pos].
        // __powf(0, 0) is NaN, so the zero exponent is taken directly.
        const int chunk_end_offset = seq_q - 1 - qbase - tid;
        const float attended_len = mass_length_gain > 0.0f ? mass_length_gain : static_cast<float>(row_pos + 1);
        mass_weight[tid] = (tid < tile_rows)
            ? (chunk_end_offset == 0 ? 1.0f : __powf(retention, static_cast<float>(chunk_end_offset))) * attended_len
            : 0.0f;
    }
    // Physical page ids of key tile t. The staging address arithmetic is a 64-bit
    // multiply chain off these, so they are read a whole iteration before the
    // stage_tile that consumes them. A tile may reach past the sequence's last
    // page; seq_table is only populated up to it.
    auto load_pages = [&](int t, int* pages) {
#pragma unroll
        for (int p = 0; p < KEY_PAGES; ++p) {
            const int page_base = t * KEY_TILE + p * PAGE_SIZE;
            pages[p] = (t < n_key_tiles && page_base < ctx_len) ? seq_table[t * KEY_PAGES + p] : 0;
        }
    };

    // Stage one key tile of the pool into its shared tile. Always commits a group,
    // empty tile included, so every iteration costs the same number of groups and a
    // wait count means the same thing throughout. `pool` selects K or V; both land
    // at the same offsets, so one routine serves either.
    auto stage_tile = [&](const scalar_t* pool, scalar_t* dst, int t, const int* pages) {
        if (t < n_key_tiles) {
#pragma unroll
            for (int p = 0; p < KEY_PAGES; ++p) {
                const int page_base = t * KEY_TILE + p * PAGE_SIZE;
                const int valid = min(PAGE_SIZE, ctx_len - page_base);
                const int64_t slot0 = (static_cast<int64_t>(pages[p]) * PAGE_SIZE) * n_kv_heads + g;
                for (int c = tid; c < kPageChunks; c += NWARPS * 32) {
                    const int off = c / kRowChunks;
                    const int d = (c % kRowChunks) * kChunkElems;
                    const int64_t idx = slot0 * HEAD_DIM + static_cast<int64_t>(off) * stride_slot + d;
                    attn::cp_async_16(
                        dst + attn::swizzled_offset<HEAD_DIM>(p * PAGE_SIZE + off, d),
                        pool + idx,
                        off < valid ? 16 : 0
                    );
                }
            }
        }
        attn::cp_async_commit();
    };

    // K of tile t is consumed by QKᵀ at the top of the iteration and V only by PV at
    // the bottom, so each is staged one full iteration ahead of its use: K right
    // after the QKᵀ that frees its tile, V right after that PV. Two groups are in
    // flight at every wait, oldest first, so both waits leave one outstanding.
    int tile_pages[KEY_PAGES], next_pages[KEY_PAGES];
    load_pages(0, tile_pages);
    stage_tile(k_pool, sK, 0, tile_pages);
    stage_tile(v_pool, sV, 0, tile_pages);
    __syncthreads();

    // Pass 1: online-softmax attention with per-(row,key) causal masking.
    for (int t = 0; t < n_key_tiles; ++t) {
        const int tile_base = t * KEY_TILE;
        load_pages(t + 1, next_pages);
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        stage_tile(k_pool, sK, t + 1, next_pages);

        if (tid < kSweepThreads) {
            const int pm = qpos[sweep_row];
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < kSweepCols; ++j) {
                const int c = sweep_lane * kSweepCols + j;
                if (tile_base + c <= pm) {
                    tmax = fmaxf(tmax, scale * sS[sweep_row * SCORE_STRIDE + c]);
                }
            }
#pragma unroll
            for (int off = kSweepLanes / 2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                const float newM = fmaxf(Mrow[sweep_row], tmax);
                // 0 when Mrow == -inf.
                row_rescale[sweep_row] = __expf(Mrow[sweep_row] - newM);
                Mrow[sweep_row] = newM;
            }
        }
        __syncthreads();

        // One thread per (query row, key): the exponentials run across the CTA,
        // and sS carries each row's weight to the sequential sum below.
        for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
            const int m = i / KEY_TILE, c = i % KEY_TILE;
            float& score = sS[m * SCORE_STRIDE + c];
            const float pw = tile_base + c <= qpos[m] ? __expf(scale * score - Mrow[m]) : 0.0f;
            score = pw;
            sP[i] = static_cast<scalar_t>(pw);
        }
        __syncthreads();

        // Summed as a shuffle tree over the row's group, not in key order.
        if (tid < kSweepThreads) {
            float lsum = 0.0f;
#pragma unroll
            for (int j = 0; j < kSweepCols; ++j) {
                lsum += sS[sweep_row * SCORE_STRIDE + sweep_lane * kSweepCols + j];
            }
#pragma unroll
            for (int off = kSweepLanes / 2; off > 0; off >>= 1) {
                lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                Lrow[sweep_row] = Lrow[sweep_row] * row_rescale[sweep_row] + lsum;
            }
        }
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_pv_registers<scalar_t, KEY_TILE, HEAD_DIM, BF16, NWARPS, true>(sP, sV, row_rescale, out_acc, warp);
        __syncthreads();
        stage_tile(v_pool, sV, t + 1, next_pages);
    }

    // Normalize + write o, each lane over the accumulators it owns.
    {
        const int gid = lane >> 2, tg = lane & 3;
        // A row past tile_rows has Lrow 0, and store() never reads its reciprocal.
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        auto store = [&](int row, int col, float value, float inv_denom) {
            if (row < tile_rows) {
                o[(static_cast<int64_t>(q_global_base + row) * n_q_heads + h) * HEAD_DIM + col] = static_cast<scalar_t>(
                    value * inv_denom
                );
            }
        };
#pragma unroll
        for (int slab = 0; slab < kSlabs; ++slab) {
            const int ds = warp + slab * NWARPS;
            if (ds >= HEAD_DIM / 16) {
                break;
            }
            const int cb = ds * 16;
            const float* acc = out_acc + slab * attn::kPvSlabAccums;
            store(gid, cb + tg * 2 + 0, acc[0], inv_low);
            store(gid, cb + tg * 2 + 1, acc[1], inv_low);
            store(gid + 8, cb + tg * 2 + 0, acc[2], inv_high);
            store(gid + 8, cb + tg * 2 + 1, acc[3], inv_high);
            store(gid, cb + 8 + tg * 2 + 0, acc[4], inv_low);
            store(gid, cb + 8 + tg * 2 + 1, acc[5], inv_low);
            store(gid + 8, cb + 8 + tg * 2 + 0, acc[6], inv_high);
            store(gid + 8, cb + 8 + tg * 2 + 1, acc[7], inv_high);
        }
    }

    // Stash LSE for the mass pass.
    if (tid < tile_rows) {
        LSE[tid] = Mrow[tid] + logf(Lrow[tid]);
        if (lse_global) {
            lse_global[static_cast<int64_t>(q_global_base + tid) * n_q_heads + h] = LSE[tid];
        }
    }
    __syncthreads();

    // Pass 2: recompute QKᵀ and accumulate per-key mass (summed over this tile's
    // query rows with p >= kp, this ONE q head), atomicAdd into this q head's own
    // mass_pool column at each key's physical slot. Nothing else in the kernel
    // touches mass, so a layer whose mass is never read stops here.
    if (!mass_pool) {
        return;
    }
    // Only K is re-streamed here, so one tile is in flight and QKᵀ frees the shared
    // tile for the next one.
    load_pages(0, tile_pages);
    stage_tile(k_pool, sK, 0, tile_pages);
    for (int t = 0; t < n_key_tiles; ++t) {
        const int tile_base = t * KEY_TILE;
        load_pages(t + 1, next_pages);
        attn::cp_async_wait<0>();
        __syncthreads();

        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        stage_tile(k_pool, sK, t + 1, next_pages);

        // One thread per (query row, key): the exponentials run across the CTA,
        // and sS carries each row's weighted share to the per-key sum below.
        for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
            const int m = i / KEY_TILE, c = i % KEY_TILE;
            float& score = sS[m * SCORE_STRIDE + c];
            score = tile_base + c <= qpos[m] ? mass_weight[m] * __expf(scale * score - LSE[m]) : 0.0f;
        }
        __syncthreads();

        // Summed in row order, so the accumulation is the sequential one.
        for (int c = tid; c < KEY_TILE; c += NWARPS * 32) {
            float ms = 0.0f;
            for (int m = 0; m < tile_rows; ++m) {
                ms += sS[m * SCORE_STRIDE + c];
            }
            if (ms > 0.0f) {
                // tile_pages is a register array, so the page it selects has to
                // be a compile-time index.
                int phys = tile_pages[0];
#pragma unroll
                for (int p = 1; p < KEY_PAGES; ++p) {
                    if (c / PAGE_SIZE == p) {
                        phys = tile_pages[p];
                    }
                }
                const int64_t mass_idx = (static_cast<int64_t>(phys) * PAGE_SIZE + c % PAGE_SIZE) * n_q_heads + h;
                atomicAdd(&mass_pool[mass_idx], mass_decay * ms);
            }
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < KEY_PAGES; ++p) {
            tile_pages[p] = next_pages[p];
        }
    }
}

// Static shared bytes one CTA of attn_prefill_tc_kernel needs at this tile shape.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, int KEY_PAGES> constexpr size_t prefill_smem_bytes() {
    constexpr size_t key_tile = static_cast<size_t>(PAGE_SIZE) * KEY_PAGES;
    return sizeof(scalar_t) * (16 * HEAD_DIM + 2 * key_tile * HEAD_DIM + 16 * key_tile) +
        sizeof(float) * (16 * (key_tile + 1) + 5 * 16) + sizeof(int) * 16;
}

constexpr size_t kMaxStaticSmem = 48 * 1024;

// KEY_PAGES, except for a shape whose tiles would not fit in static shared
// memory.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, int KEY_PAGES> constexpr int prefill_key_pages() {
    return prefill_smem_bytes<scalar_t, PAGE_SIZE, HEAD_DIM, KEY_PAGES>() <= kMaxStaticSmem ? KEY_PAGES : 1;
}

}  // namespace attn
}  // namespace pulsar
