#include "attn_common.cuh"

#include "pulsar/kernels/attn_tuning.hpp"
#include "pulsar/ops.hpp"
#include "pulsar/kernels/paged_prefill.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// Paged-KV prefill attention with a per-key mass side output. The prefill
// counterpart of attn_decode: multiple query tokens per sequence (seq_q > 1),
// causal-masked, over the same paged KV pool, for a ragged (varlen) batch. The
// pool's slot arithmetic is in pulsar/runtime/kv/paged_pool.hpp.
//
// Shapes (the pools are one layer's slice, as in the decode op):
//   q             [total_q, n_q_heads, head_dim]  (all seqs' queries concatenated)
//   k_pool,v_pool [num_pages, page_size, n_kv_heads, head_dim]
//   mass_pool     [num_pages, page_size, n_q_heads]  (fp32, added to)
//   page_tables   int32 [num_seqs, max_pages]
//   cu_seqlens_q  int32 [num_seqs+1]  prefix sums of per-seq query lengths
//   seqlens_k     int32 [num_seqs]    total context length per sequence
//   attention_mass_decay fp32 [num_seqs]  per-sequence EMA gain alpha
//   mass_length_gain                      length each query's mass is stated against;
//                                         <= 0 uses that query's own causal key count
//
// For sequence i, seq_q_i = cu_seqlens_q[i+1] - cu_seqlens_q[i]; its query tokens
// occupy context pos [ctx_start_i, seqlens_k[i]) with ctx_start_i =
// seqlens_k[i] - seq_q_i (>= 0). Query token at context pos p attends to keys at
// context pos [0, p] inclusive. mass at key kp for query head h is the sum, over
// that head's query tokens with p >= kp, of the normalized softmax weight (per
// query head, no group sum; same units as decode).

namespace pulsar {
namespace {

using attn::kMaxHeadDim;
using attn::kThreads;

// Scalar reference / fallback kernel. One CTA per (global query token, q head);
// threads stride over the causal key range [0, p] running an online softmax, then
// a second streaming pass atomicAdds the per-key normalized mass. The decode
// scalar kernel with ctx_len replaced by the causal bound (p+1) and a per-CTA
// sequence lookup (binary search over cu_seqlens_q).
template <typename scalar_t, int PAGE_SIZE>
__global__ void paged_prefill_scalar_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_pool,
    const scalar_t* __restrict__ v_pool,
    float* __restrict__ mass_pool,  // in-place +=; null skips the mass pass
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seqlens_k,
    const float* __restrict__ attention_mass_decay,  // [num_seqs] EMA gain alpha
    float mass_length_gain,  // <= 0 => each query's own causal key count
    scalar_t* __restrict__ o,
    float* __restrict__ lse_out,  // [total_q, n_q_heads]; null when not captured
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int head_dim,
    int group,
    float scale,
    int num_seqs
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int qtok = blk / n_q_heads;  // global query-token index
    const int h = blk % n_q_heads;
    const int g = h / group;

    // Sequence owning this query token: cu_seqlens_q[s] <= qtok < cu_seqlens_q[s+1].
    int lo = 0, hi = num_seqs;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (cu_seqlens_q[mid] <= qtok) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int s = lo;
    const int q0 = cu_seqlens_q[s];
    const int seq_q = cu_seqlens_q[s + 1] - q0;
    const int ctx_len = seqlens_k[s];
    const int ctx_start = ctx_len - seq_q;
    const int p = ctx_start + (qtok - q0);  // this query's context pos
    const int key_count = p + 1;  // causal keys [0, p]
    // This sequence's EMA gain alpha and the retention 1 - alpha it implies.
    const float mass_decay = attention_mass_decay[s];
    const float retention = 1.0f - mass_decay;
    // The length the mass is stated against: the keys THIS query attends, or the
    // caller's own gain in their place.
    const float attended_len = mass_length_gain > 0.0f ? mass_length_gain : static_cast<float>(key_count);
    // Per-query multiplier on the mass this query hands out: the within-chunk
    // retention grading (offset from the chunk end, retention^0 for the last query in
    // the chunk, so a key's mass is the per-token EMA; retention == 1 -> 1) times that
    // length.
    // __powf(0, 0) is NaN, so the zero exponent is taken directly.
    const int chunk_end_offset = seq_q - 1 - (qtok - q0);
    const float mass_weight = (chunk_end_offset == 0 ? 1.0f : __powf(retention, static_cast<float>(chunk_end_offset))) *
        attended_len;

    const int32_t* seq_table = page_tables + static_cast<int64_t>(s) * max_pages;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = q + (static_cast<int64_t>(qtok) * n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads) {
        qsh[d] = static_cast<float>(qrow[d]);
    }
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    for (int j = tid; j < key_count; j += kThreads) {
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
        const float corr = __expf(m - new_m);
        const float pw = __expf(sc - new_m);
        l = l * corr + pw;
        const scalar_t* vrow = v_pool + base;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + pw * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    // Combine per-thread states: global row max, then denom-weighted output.
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
        lse_out[static_cast<int64_t>(qtok) * n_q_heads + h] = M + logf(denom);
    }
    scalar_t* orow = o + (static_cast<int64_t>(qtok) * n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    // Second streaming pass: normalized weight per causal key, atomicAdd into
    // mass_pool at the key's physical slot, THIS query head's own column h (per query
    // head, no group sum; accumulates across query tokens with p >= kp).
    if (!mass_pool) {
        return;
    }
    for (int j = tid; j < key_count; j += kThreads) {
        const int phys = seq_table[j / PAGE_SIZE];
        const int offset = j % PAGE_SIZE;
        const int64_t base = ((static_cast<int64_t>(phys) * PAGE_SIZE + offset) * n_kv_heads + g) * head_dim;
        const scalar_t* krow = k_pool + base;
        float sc = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            sc += qsh[d] * static_cast<float>(krow[d]);
        }
        sc *= scale;
        const float pw = __expf(sc - M) * inv_denom;
        const int64_t mass_idx = (static_cast<int64_t>(phys) * PAGE_SIZE + offset) * n_q_heads + h;
        atomicAdd(&mass_pool[mass_idx], mass_decay * pw * mass_weight);
    }
}

// Instantiate + launch the tensor-core kernel for the runtime (page_size,
// head_dim) at the launch shape this device's tuning row names.
template <typename scalar_t, bool BF16>
void launch_tc_prefill(
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    float* mass,
    const int32_t* bt,
    const int32_t* cu,
    const int32_t* sk,
    const float* decay,
    float mass_length_gain,
    const int32_t* tseq,
    const int32_t* tqb,
    scalar_t* o,
    float* lse_global,
    int n_q_heads,
    int n_kv_heads,
    int max_pages,
    int group,
    int head_dim,
    int page_size,
    float scale,
    int64_t grid,
    cudaStream_t stream
) {
    attn::dispatch_page_head("attn_prefill", page_size, head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr attn::AttnTuning kTuning = attn::kAttnTuningTable[decltype(row_tag)::value];
            constexpr int NWARPS = kTuning.prefill_warps;
            attn::attn_prefill_tc_kernel<
                scalar_t,
                BS,
                HD,
                BF16,
                NWARPS,
                attn::prefill_key_pages<scalar_t, BS, HD, kTuning.prefill_key_pages>(),
                kTuning.prefill_ctas_per_sm><<<static_cast<int>(grid), NWARPS * 32, 0, stream>>>(
                q,
                k,
                v,
                mass,
                bt,
                cu,
                sk,
                decay,
                mass_length_gain,
                tseq,
                tqb,
                o,
                lse_global,
                n_q_heads,
                n_kv_heads,
                max_pages,
                group,
                scale
            );
        });
    });
}

// Shared body of the tensor-core op and its scalar-only reference sibling.
// force_scalar routes everything through the scalar kernel; the default op uses
// the tensor-core kernel for fp16/bf16 with head_dim in {64,128} and page_size in
// {16,32}, and falls back to the scalar kernel otherwise.
at::Tensor paged_prefill_impl(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain,
    bool force_scalar,
    const at::Tensor& lse_capture = {}
) {
    TORCH_CHECK(
        !lse_capture.defined() ||
            (lse_capture.is_cuda() && lse_capture.scalar_type() == at::kFloat && lse_capture.is_contiguous()),
        "attn_prefill: lse capture must be a contiguous fp32 CUDA tensor"
    );
    float* lse_ptr = lse_capture.defined() ? lse_capture.data_ptr<float>() : nullptr;
    // An undefined mass_pool states that no reader narrows to this layer, and the
    // kernels then skip the mass pass entirely.
    const bool accumulate_mass = mass_pool.defined();
    TORCH_CHECK(
        q.is_cuda() && k_pool.is_cuda() && v_pool.is_cuda() && (!accumulate_mass || mass_pool.is_cuda()) &&
            page_tables.is_cuda() && cu_seqlens_q.is_cuda() && seqlens_k.is_cuda() && attention_mass_decay.is_cuda(),
        "attn_prefill: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        attention_mass_decay.scalar_type() == at::kFloat && attention_mass_decay.dim() == 1,
        "attn_prefill: attention_mass_decay must be fp32 1-D [num_seqs]"
    );
    TORCH_CHECK(
        q.scalar_type() == k_pool.scalar_type() && k_pool.scalar_type() == v_pool.scalar_type(),
        "attn_prefill: q, k_pool, v_pool must share a dtype"
    );
    TORCH_CHECK(!accumulate_mass || mass_pool.scalar_type() == at::kFloat, "attn_prefill: mass_pool must be fp32");
    TORCH_CHECK(
        page_tables.scalar_type() == at::kInt && cu_seqlens_q.scalar_type() == at::kInt &&
            seqlens_k.scalar_type() == at::kInt,
        "attn_prefill: page_tables, cu_seqlens_q, seqlens_k must be int32"
    );
    TORCH_CHECK(q.dim() == 3, "attn_prefill: q must be 3-D [total_q, n_q_heads, head_dim]");
    TORCH_CHECK(
        k_pool.dim() == 4 && v_pool.dim() == 4,
        "attn_prefill: pools must be 4-D [num_pages, page_size, n_kv_heads, head_dim]"
    );
    TORCH_CHECK(
        !accumulate_mass || mass_pool.dim() == 3,
        "attn_prefill: mass_pool must be 3-D [num_pages, page_size, n_q_heads]"
    );
    TORCH_CHECK(page_tables.dim() == 2, "attn_prefill: page_tables must be 2-D [num_seqs, max_pages]");
    TORCH_CHECK(
        cu_seqlens_q.dim() == 1 && seqlens_k.dim() == 1,
        "attn_prefill: cu_seqlens_q and seqlens_k must be 1-D"
    );

    const int64_t total_q = q.size(0);
    const int64_t n_q_heads = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t num_pages = k_pool.size(0);
    const int64_t page_size = k_pool.size(1);
    const int64_t n_kv_heads = k_pool.size(2);
    const int64_t max_pages = page_tables.size(1);
    const int64_t num_seqs = seqlens_k.size(0);

    TORCH_CHECK(
        v_pool.size(0) == num_pages && v_pool.size(1) == page_size && v_pool.size(2) == n_kv_heads &&
            v_pool.size(3) == head_dim,
        "attn_prefill: v_pool must match k_pool shape"
    );
    TORCH_CHECK(
        !accumulate_mass ||
            (mass_pool.size(0) == num_pages && mass_pool.size(1) == page_size && mass_pool.size(2) == n_q_heads),
        "attn_prefill: mass_pool must be [num_pages, page_size, n_q_heads]"
    );
    TORCH_CHECK(k_pool.size(3) == head_dim, "attn_prefill: pool head_dim must match q");
    TORCH_CHECK(
        page_tables.size(0) == num_seqs && cu_seqlens_q.size(0) == num_seqs + 1,
        "attn_prefill: page_tables/cu_seqlens_q must match num_seqs"
    );
    TORCH_CHECK(
        attention_mass_decay.size(0) == num_seqs,
        "attn_prefill: attention_mass_decay has ",
        attention_mass_decay.size(0),
        " entries for ",
        num_seqs,
        " sequences"
    );
    TORCH_CHECK(
        n_kv_heads > 0 && n_q_heads % n_kv_heads == 0,
        "attn_prefill: n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(head_dim <= kMaxHeadDim, "attn_prefill: head_dim exceeds compile-time max");
    TORCH_CHECK(
        k_pool.is_contiguous() && v_pool.is_contiguous() && (!accumulate_mass || mass_pool.is_contiguous()),
        "attn_prefill: pools must be contiguous"
    );

    auto qc = q.contiguous();
    auto bt = page_tables.contiguous();
    auto cu = cu_seqlens_q.contiguous();
    auto sk = seqlens_k.contiguous();
    auto decay = attention_mass_decay.contiguous();
    auto o = at::empty_like(qc);

    if (total_q == 0 || num_seqs == 0) {
        return o;
    }

    const int group = static_cast<int>(n_q_heads / n_kv_heads);
    float* mass_ptr = accumulate_mass ? mass_pool.data_ptr<float>() : nullptr;
    const float gain = static_cast<float>(mass_length_gain);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto dt = qc.scalar_type();
    const bool tc_ok = !force_scalar && (dt == at::kHalf || dt == at::kBFloat16) &&
        (head_dim == 64 || head_dim == 128) && (page_size == 16 || page_size == 32);

    if (tc_ok) {
        // Build the (seq, query-tile-base) work list on the host: each query tile
        // is up to 16 query tokens of one sequence. Grid = ntiles * n_q_heads.
        auto cu_cpu = cu.to(at::kCPU);
        const int32_t* cup = cu_cpu.data_ptr<int32_t>();
        std::vector<int32_t> h_tseq, h_tqb;
        for (int64_t s = 0; s < num_seqs; ++s) {
            const int32_t sq = cup[s + 1] - cup[s];
            for (int32_t qb = 0; qb < sq; qb += 16) {
                h_tseq.push_back(static_cast<int32_t>(s));
                h_tqb.push_back(qb);
            }
        }
        const int64_t ntiles = static_cast<int64_t>(h_tseq.size());
        if (ntiles == 0) {
            return o;
        }

        auto iopts = at::TensorOptions().dtype(at::kInt);
        auto tseq = at::from_blob(h_tseq.data(), {ntiles}, iopts).to(qc.device());
        auto tqb = at::from_blob(h_tqb.data(), {ntiles}, iopts).to(qc.device());
        const int64_t grid = ntiles * n_q_heads;

        if (dt == at::kBFloat16) {
            launch_tc_prefill<at::BFloat16, true>(
                qc.data_ptr<at::BFloat16>(),
                k_pool.data_ptr<at::BFloat16>(),
                v_pool.data_ptr<at::BFloat16>(),
                mass_ptr,
                bt.data_ptr<int32_t>(),
                cu.data_ptr<int32_t>(),
                sk.data_ptr<int32_t>(),
                decay.data_ptr<float>(),
                gain,
                tseq.data_ptr<int32_t>(),
                tqb.data_ptr<int32_t>(),
                o.data_ptr<at::BFloat16>(),
                lse_ptr,
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                group,
                static_cast<int>(head_dim),
                static_cast<int>(page_size),
                static_cast<float>(scale),
                grid,
                stream
            );
        } else {
            launch_tc_prefill<at::Half, false>(
                qc.data_ptr<at::Half>(),
                k_pool.data_ptr<at::Half>(),
                v_pool.data_ptr<at::Half>(),
                mass_ptr,
                bt.data_ptr<int32_t>(),
                cu.data_ptr<int32_t>(),
                sk.data_ptr<int32_t>(),
                decay.data_ptr<float>(),
                gain,
                tseq.data_ptr<int32_t>(),
                tqb.data_ptr<int32_t>(),
                o.data_ptr<at::Half>(),
                lse_ptr,
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                group,
                static_cast<int>(head_dim),
                static_cast<int>(page_size),
                static_cast<float>(scale),
                grid,
                stream
            );
        }
        return o;
    }

    const int64_t blocks = total_q * n_q_heads;
    const size_t smem = static_cast<size_t>(2 * head_dim + kThreads) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qc.scalar_type(), "attn_prefill_cuda", [&] {
        attn::dispatch_page_size("attn_prefill", page_size, [&](auto page_size_tag) {
            constexpr int BS = decltype(page_size_tag)::value;
            paged_prefill_scalar_kernel<scalar_t, BS><<<static_cast<int>(blocks), kThreads, smem, stream>>>(
                qc.data_ptr<scalar_t>(),
                k_pool.data_ptr<scalar_t>(),
                v_pool.data_ptr<scalar_t>(),
                mass_ptr,
                bt.data_ptr<int32_t>(),
                cu.data_ptr<int32_t>(),
                sk.data_ptr<int32_t>(),
                decay.data_ptr<float>(),
                gain,
                o.data_ptr<scalar_t>(),
                lse_ptr,
                static_cast<int>(n_q_heads),
                static_cast<int>(n_kv_heads),
                static_cast<int>(max_pages),
                static_cast<int>(head_dim),
                group,
                static_cast<float>(scale),
                static_cast<int>(num_seqs)
            );
        });
    });
    return o;
}

}  // namespace

at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    double scale,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture,
    double mass_length_gain
) {
    return paged_prefill_impl(
        q,
        k_pool,
        v_pool,
        mass_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        scale,
        attention_mass_decay,
        mass_length_gain,
        /*force_scalar=*/false,
        lse_capture.has_value() ? *lse_capture : at::Tensor{}
    );
}

at::Tensor attn_prefill_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain
) {
    return paged_prefill_impl(
        q,
        k_pool,
        v_pool,
        mass_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        scale,
        attention_mass_decay,
        mass_length_gain,
        /*force_scalar=*/true
    );
}

}  // namespace pulsar
