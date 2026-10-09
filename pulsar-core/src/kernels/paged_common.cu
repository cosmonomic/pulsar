#include "attn_common.cuh"

#include "pulsar/rope.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>

// The argument contract both paged attention ops share, and the scalar kernel both
// fall back to. Decode is the case of one query per sequence at the context's last
// position, so one kernel serves both.

namespace pulsar {
namespace attn {
namespace {

// One CTA per (query token, query head). Threads stride over the causal key range
// [0, p] running independent online softmaxes, combined with the flash rescale; a
// second streaming pass recomputes each normalized weight and adds it into the mass.
template <typename scalar_t, int PAGE_SIZE> __global__ void attn_scalar_kernel(const PagedAttnParams<scalar_t> p) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int qtok = blk / p.n_q_heads;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    const int head_dim = p.head_dim;
    const int half = head_dim / 2;

    int s = qtok;
    if (p.cu_seqlens_q) {
        int lo = 0, hi = p.num_seqs;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (p.cu_seqlens_q[mid] <= qtok) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = lo;
    }
    const int q0 = p.cu_seqlens_q ? p.cu_seqlens_q[s] : s;
    const int seq_q = p.cu_seqlens_q ? p.cu_seqlens_q[s + 1] - q0 : 1;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const int key_count = query_pos + 1;
    const KeyLayout layout = key_layout(p.rope_layout, s);
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads) {
        qsh[d] = static_cast<float>(qrow[d]);
    }
    __syncthreads();

    auto key_offset = [&](int j) {
        const int line = seq_table[j / PAGE_SIZE];
        return ((static_cast<int64_t>(line) * PAGE_SIZE + j % PAGE_SIZE) * p.n_kv_heads + g) * head_dim;
    };
    auto score = [&](int j, int64_t offset) {
        const scalar_t* krow = p.k_pool + offset;
        const int pos = layout.position(j);
        float sc = 0.0f;
        for (int i = 0; i < half; ++i) {
            float cos_a, sin_a;
            rope_sincos_turns(pos, p.rope_turns[i], cos_a, sin_a);
            const auto k = rope_rotate<scalar_t>(
                static_cast<float>(krow[i]),
                static_cast<float>(krow[i + half]),
                cos_a,
                sin_a
            );
            sc += qsh[i] * static_cast<float>(k.lo) + qsh[i + half] * static_cast<float>(k.hi);
        }
        return sc * p.scale;
    };

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    for (int j = tid; j < key_count; j += kThreads) {
        const int64_t offset = key_offset(j);
        const float sc = score(j, offset);
        const float new_m = fmaxf(m, sc);
        const float corr = __expf(m - new_m);  // 0 when m == -inf
        const float pw = __expf(sc - new_m);
        l = l * corr + pw;
        const scalar_t* vrow = p.v_pool + offset;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + pw * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    const float M = block_reduce_max<kThreads>(red, tid, m);
    const float w = __expf(m - M);  // 0 when this thread saw no keys
    for (int d = tid; d < head_dim; d += kThreads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = block_reduce_sum<kThreads>(red, tid, l * w);

    if (p.lse && tid == 0) {
        p.lse[static_cast<int64_t>(qtok) * p.n_q_heads + h] = M + logf(denom);
    }
    scalar_t* orow = p.o + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    if (!p.mass) {
        return;
    }
    // The within-chunk retention grades a query by its offset from the chunk end, so a
    // key's mass is the per-token EMA of the attention it received; the length gain
    // states the weight against the keys this query attended. __powf(0, 0) is NaN, so
    // the zero exponent is taken directly.
    const float mass_decay = p.mass_decay[s];
    const int chunk_end_offset = seq_q - 1 - (qtok - q0);
    const float graded = chunk_end_offset == 0 ? 1.0f : __powf(1.0f - mass_decay, static_cast<float>(chunk_end_offset));
    const float attended_len = p.mass_length_gain > 0.0f ? p.mass_length_gain : static_cast<float>(key_count);
    const float mass_weight = mass_decay * graded * attended_len;
    float* mass = seq_mass(p, s, ctx_len, PAGE_SIZE);
    for (int j = tid; j < key_count; j += kThreads) {
        const float pw = __expf(score(j, key_offset(j)) - M) * inv_denom;
        atomicAdd(&mass[static_cast<int64_t>(j) * p.n_q_heads + h], mass_weight * pw);
    }
}

}  // namespace

bool PagedAttnBatch::tensor_core_eligible() const {
    const auto dt = this->q.scalar_type();
    return (dt == at::kHalf || dt == at::kBFloat16) && (this->head_dim == 64 || this->head_dim == 128) &&
        (this->page_size == 16 || this->page_size == 32);
}

PagedAttnBatch paged_attn_batch(
    const char* op,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& rope_layout,
    double rope_theta,
    double scale,
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& attention_mass_decay,
    double mass_length_gain,
    const std::optional<at::Tensor>& lse_capture
) {
    auto on_cuda = [](const at::Tensor& t) { return t.is_cuda(); };
    auto int32_dims = [](const at::Tensor& t, int64_t dims) { return t.scalar_type() == at::kInt && t.dim() == dims; };

    TORCH_CHECK(
        on_cuda(q) && on_cuda(k_pool) && on_cuda(v_pool) && on_cuda(page_tables) && on_cuda(seqlens_k) &&
            on_cuda(rope_layout),
        op,
        ": all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        q.scalar_type() == k_pool.scalar_type() && k_pool.scalar_type() == v_pool.scalar_type(),
        op,
        ": q, k_pool, v_pool must share a dtype"
    );
    TORCH_CHECK(q.dim() == 3, op, ": q must be 3-D [total_q, n_q_heads, head_dim]");
    TORCH_CHECK(
        k_pool.dim() == 4 && k_pool.sizes() == v_pool.sizes(),
        op,
        ": k_pool and v_pool must be 4-D [num_lines, page_size, n_kv_heads, head_dim] of one shape"
    );
    TORCH_CHECK(k_pool.is_contiguous() && v_pool.is_contiguous(), op, ": pools must be contiguous");

    PagedAttnBatch b;
    b.total_q = q.size(0);
    b.n_q_heads = q.size(1);
    b.head_dim = q.size(2);
    b.page_size = k_pool.size(1);
    b.n_kv_heads = k_pool.size(2);
    b.num_seqs = seqlens_k.size(0);
    b.scale = scale;
    b.mass_length_gain = mass_length_gain;

    TORCH_CHECK(k_pool.size(3) == b.head_dim, op, ": pool head_dim must match q");
    TORCH_CHECK(b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim, op, ": head_dim must be even and <= ", kMaxHeadDim);
    TORCH_CHECK(
        b.n_kv_heads > 0 && b.n_q_heads % b.n_kv_heads == 0,
        op,
        ": n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(int32_dims(seqlens_k, 1), op, ": seqlens_k must be int32 [num_seqs]");
    TORCH_CHECK(
        int32_dims(page_tables, 2) && page_tables.size(0) == b.num_seqs,
        op,
        ": page_tables must be int32 [num_seqs, max_pages]"
    );
    TORCH_CHECK(
        int32_dims(rope_layout, 2) && rope_layout.size(0) == b.num_seqs && rope_layout.size(1) == 3,
        op,
        ": rope_layout must be int32 [num_seqs, 3]"
    );
    if (cu_seqlens_q.has_value()) {
        TORCH_CHECK(
            on_cuda(*cu_seqlens_q) && int32_dims(*cu_seqlens_q, 1) && cu_seqlens_q->size(0) == b.num_seqs + 1,
            op,
            ": cu_seqlens_q must be int32 [num_seqs + 1] on CUDA"
        );
        b.cu_seqlens_q = cu_seqlens_q->contiguous();
    } else {
        TORCH_CHECK(b.total_q == b.num_seqs, op, ": one query row per sequence");
    }
    b.max_pages = page_tables.size(1);

    const bool accumulates = mass.has_value();
    TORCH_CHECK(
        cu_view_pages.has_value() == accumulates && attention_mass_decay.has_value() == accumulates,
        op,
        ": mass, cu_view_pages and attention_mass_decay are passed together or not at all"
    );
    if (accumulates) {
        TORCH_CHECK(
            on_cuda(*mass) && on_cuda(*cu_view_pages) && on_cuda(*attention_mass_decay),
            op,
            ": all inputs must be CUDA tensors"
        );
        TORCH_CHECK(
            mass->scalar_type() == at::kFloat && mass->dim() == 3 && mass->size(1) == b.page_size &&
                mass->size(2) == b.n_q_heads && mass->is_contiguous(),
            op,
            ": mass must be contiguous fp32 [total_view_pages, page_size, n_q_heads]"
        );
        TORCH_CHECK(
            int32_dims(*cu_view_pages, 1) && cu_view_pages->size(0) == b.num_seqs + 1,
            op,
            ": cu_view_pages must be int32 [num_seqs + 1]"
        );
        TORCH_CHECK(
            attention_mass_decay->scalar_type() == at::kFloat && attention_mass_decay->dim() == 1 &&
                attention_mass_decay->size(0) == b.num_seqs,
            op,
            ": attention_mass_decay must be fp32 [num_seqs]"
        );
        b.mass = *mass;
        b.cu_view_pages = cu_view_pages->contiguous();
        b.mass_decay = attention_mass_decay->contiguous();
    }
    if (lse_capture.has_value()) {
        TORCH_CHECK(
            on_cuda(*lse_capture) && lse_capture->scalar_type() == at::kFloat && lse_capture->is_contiguous() &&
                lse_capture->numel() == b.total_q * b.n_q_heads,
            op,
            ": lse_capture must be a contiguous fp32 CUDA tensor of [total_q, n_q_heads] elements"
        );
        b.lse = *lse_capture;
    }

    b.q = q.contiguous();
    b.k_pool = k_pool;
    b.v_pool = v_pool;
    b.page_tables = page_tables.contiguous();
    b.seqlens_k = seqlens_k.contiguous();
    b.rope_layout = rope_layout.contiguous();
    b.rope_turns = rope_turns_per_position(rope_theta, b.head_dim, q.device());
    b.o = at::empty_like(b.q);
    return b;
}

void launch_attn_scalar(const PagedAttnBatch& batch, cudaStream_t stream) {
    if (batch.total_q == 0 || batch.num_seqs == 0) {
        return;
    }
    const int64_t blocks = batch.total_q * batch.n_q_heads;
    const size_t smem = static_cast<size_t>(2 * batch.head_dim + kThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, batch.q.scalar_type(), "attn_scalar", [&] {
        dispatch_page_size("attn_scalar", batch.page_size, [&](auto page_size_tag) {
            constexpr int PS = decltype(page_size_tag)::value;
            attn_scalar_kernel<scalar_t, PS>
                <<<static_cast<int>(blocks), kThreads, smem, stream>>>(batch.params<scalar_t>());
        });
    });
}

}  // namespace attn
}  // namespace pulsar
