#include "attn_common.cuh"

#include "pulsar/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <math_constants.h>

#include <tuple>

// Fused causal attention over contiguous (unpaged) K/V, with a per-key
// attention_mass side output.
//
//   o[h,i]        = sum_j softmax_j(scale * q[h,i]·k[g,j]) * v[g,j]
//   attention_mass[g] += sum over (h->g, i) of the normalized softmax weight p[i,j]
//
// where g = h / (n_q_heads / n_kv_heads) is the GQA-shared kv head, and key j is
// allowed for query row i iff j <= causal_offset + i.
//
// MASS SEMANTICS DIFFER FROM THE PAGED KERNELS. Here mass is indexed by KV head
// and SUMMED over the query heads of the group; paged_attention.cu and
// paged_prefill.cu index mass by QUERY head and never sum across a group. The two
// are not interchangeable signals.
//
// One block handles one (query head h, query row i). Threads stride over the
// allowed keys; each thread runs an independent numerically-stable online
// ("flash") softmax over its own subset of keys, keeping a running max, running
// denominator, and a running head_dim output accumulator. The full [seq_q, L]
// score matrix is never materialized: a thread holds only one query row's state.
// A second streaming pass recomputes each score once the row's max/denominator
// are known and atomicAdds the normalized weight into attention_mass. Everything
// accumulates in fp32.

namespace pulsar {
namespace {

using attn::kMaxHeadDim;
using attn::kThreads;

template <typename scalar_t>
__global__ void attn_causal_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    float* __restrict__ attention_mass,  // in-place +=
    scalar_t* __restrict__ o,
    int seq_q,
    int64_t L,
    int head_dim,
    int group,
    float scale,
    int64_t causal_offset
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int h = blk / seq_q;
    const int i = blk % seq_q;
    const int g = h / group;

    // Inclusive index of the last key this query row may attend to.
    int64_t last_key = causal_offset + i;
    if (last_key > L - 1) {
        last_key = L - 1;
    }

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = q + (static_cast<int64_t>(h) * seq_q + i) * head_dim;
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

    const scalar_t* kbase = k + static_cast<int64_t>(g) * L * head_dim;
    const scalar_t* vbase = v + static_cast<int64_t>(g) * L * head_dim;

    for (int64_t j = tid; j <= last_key; j += kThreads) {
        const scalar_t* krow = kbase + j * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += qsh[d] * static_cast<float>(krow[d]);
        }
        s *= scale;

        const float new_m = fmaxf(m, s);
        const float corr = __expf(m - new_m);  // 0 when m == -inf
        const float p = __expf(s - new_m);
        l = l * corr + p;
        const scalar_t* vrow = vbase + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + p * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    // Combine per-thread states with the flash rescale: weight each thread's
    // contribution by exp(m_t - M) into the denominator and the output vector.
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

    scalar_t* orow = o + (static_cast<int64_t>(h) * seq_q + i) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    // Second streaming pass: recompute each score and atomicAdd the normalized
    // weight into the shared attention_mass signal (accumulates across the query
    // heads and rows that map to this kv head).
    for (int64_t j = tid; j <= last_key; j += kThreads) {
        const scalar_t* krow = kbase + j * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += qsh[d] * static_cast<float>(krow[d]);
        }
        s *= scale;
        const float p = __expf(s - M) * inv_denom;
        atomicAdd(&attention_mass[static_cast<int64_t>(g) * L + j], p);
    }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> attn_causal_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& attention_mass,
    double scale,
    int64_t causal_offset
) {
    TORCH_CHECK(
        q.is_cuda() && k.is_cuda() && v.is_cuda() && attention_mass.is_cuda(),
        "attn_causal: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
        "attn_causal: q, k, v must share a dtype"
    );
    TORCH_CHECK(attention_mass.scalar_type() == at::kFloat, "attn_causal: attention_mass must be fp32");
    TORCH_CHECK(
        q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
        "attn_causal: q, k, v must be 3-D [heads, len, head_dim]"
    );
    TORCH_CHECK(attention_mass.dim() == 2, "attn_causal: attention_mass must be 2-D [kv_heads, L]");

    const int64_t n_q_heads = q.size(0);
    const int64_t seq_q = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t n_kv_heads = k.size(0);
    const int64_t L = k.size(1);

    TORCH_CHECK(
        v.size(0) == n_kv_heads && v.size(1) == L && v.size(2) == head_dim,
        "attn_causal: v must match k shape [kv_heads, L, head_dim]"
    );
    TORCH_CHECK(k.size(2) == head_dim, "attn_causal: k head_dim must match q");
    TORCH_CHECK(
        attention_mass.size(0) == n_kv_heads && attention_mass.size(1) == L,
        "attn_causal: attention_mass must be [kv_heads, L]"
    );
    TORCH_CHECK(
        n_kv_heads > 0 && n_q_heads % n_kv_heads == 0,
        "attn_causal: n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(head_dim <= kMaxHeadDim, "attn_causal: head_dim exceeds compile-time max");

    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();

    auto o = at::empty_like(qc);
    // attention_mass_out = attention_mass_in + this call's contribution; start from a
    // copy and let the kernel atomicAdd the contribution in place.
    auto attention_mass_out = attention_mass.contiguous().clone();

    const int group = static_cast<int>(n_q_heads / n_kv_heads);
    const int64_t blocks = n_q_heads * seq_q;
    const size_t smem = static_cast<size_t>(2 * head_dim + kThreads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qc.scalar_type(), "attn_causal_cuda", [&] {
        attn_causal_kernel<scalar_t><<<blocks, kThreads, smem, stream>>>(
            qc.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            vc.data_ptr<scalar_t>(),
            attention_mass_out.data_ptr<float>(),
            o.data_ptr<scalar_t>(),
            static_cast<int>(seq_q),
            L,
            static_cast<int>(head_dim),
            group,
            static_cast<float>(scale),
            causal_offset
        );
    });
    return std::make_tuple(o, attention_mass_out);
}

namespace {

// Cache-form variant of the kernel above. The K/V/mass buffers have a fixed row
// capacity max_len while only rows [0, valid_len) are valid, so the key loop and
// the causal clamp are bounded by valid_len while the row stride stays max_len.
// The mass buffer is mutated in place (no clone).
template <typename scalar_t>
__global__ void attn_causal_cache_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    float* __restrict__ attention_mass,  // in-place +=
    scalar_t* __restrict__ o,
    int seq_q,
    int64_t max_len,
    int64_t valid_len,
    int head_dim,
    int group,
    float scale,
    int64_t causal_offset
) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int h = blk / seq_q;
    const int i = blk % seq_q;
    const int g = h / group;

    int64_t last_key = causal_offset + i;
    if (last_key > valid_len - 1) {
        last_key = valid_len - 1;
    }

    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = q + (static_cast<int64_t>(h) * seq_q + i) * head_dim;
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

    const scalar_t* kbase = k + static_cast<int64_t>(g) * max_len * head_dim;
    const scalar_t* vbase = v + static_cast<int64_t>(g) * max_len * head_dim;

    for (int64_t j = tid; j <= last_key; j += kThreads) {
        const scalar_t* krow = kbase + j * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += qsh[d] * static_cast<float>(krow[d]);
        }
        s *= scale;

        const float new_m = fmaxf(m, s);
        const float corr = __expf(m - new_m);
        const float p = __expf(s - new_m);
        l = l * corr + p;
        const scalar_t* vrow = vbase + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + p * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    const float M = attn::block_reduce_max<kThreads>(red, tid, m);
    const float w = __expf(m - M);
    for (int d = tid; d < head_dim; d += kThreads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = attn::block_reduce_sum<kThreads>(red, tid, l * w);

    scalar_t* orow = o + (static_cast<int64_t>(h) * seq_q + i) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    for (int64_t j = tid; j <= last_key; j += kThreads) {
        const scalar_t* krow = kbase + j * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            s += qsh[d] * static_cast<float>(krow[d]);
        }
        s *= scale;
        const float p = __expf(s - M) * inv_denom;
        atomicAdd(&attention_mass[static_cast<int64_t>(g) * max_len + j], p);
    }
}

}  // namespace

at::Tensor attn_causal_cache_cuda(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& attention_mass,
    double scale,
    int64_t cur_len
) {
    TORCH_CHECK(
        q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda() && attention_mass.is_cuda(),
        "attn_causal_cache: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        q.scalar_type() == k_cache.scalar_type() && k_cache.scalar_type() == v_cache.scalar_type(),
        "attn_causal_cache: q, k_cache, v_cache must share a dtype"
    );
    TORCH_CHECK(attention_mass.scalar_type() == at::kFloat, "attn_causal_cache: attention_mass must be fp32");
    TORCH_CHECK(
        q.dim() == 3 && k_cache.dim() == 3 && v_cache.dim() == 3,
        "attn_causal_cache: q, k_cache, v_cache must be 3-D [heads, len, head_dim]"
    );
    TORCH_CHECK(attention_mass.dim() == 2, "attn_causal_cache: attention_mass must be 2-D [kv_heads, max_len]");

    const int64_t n_q_heads = q.size(0);
    const int64_t seq_q = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t n_kv_heads = k_cache.size(0);
    const int64_t max_len = k_cache.size(1);
    const int64_t valid_len = cur_len + seq_q;

    TORCH_CHECK(
        v_cache.size(0) == n_kv_heads && v_cache.size(1) == max_len && v_cache.size(2) == head_dim,
        "attn_causal_cache: v_cache must match k_cache shape [kv_heads, max_len, "
        "head_dim]"
    );
    TORCH_CHECK(k_cache.size(2) == head_dim, "attn_causal_cache: k_cache head_dim must match q");
    TORCH_CHECK(
        attention_mass.size(0) == n_kv_heads && attention_mass.size(1) == max_len,
        "attn_causal_cache: attention_mass must be [kv_heads, max_len]"
    );
    TORCH_CHECK(
        n_kv_heads > 0 && n_q_heads % n_kv_heads == 0,
        "attn_causal_cache: n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(head_dim <= kMaxHeadDim, "attn_causal_cache: head_dim exceeds compile-time max");
    TORCH_CHECK(cur_len >= 0, "attn_causal_cache: cur_len must be non-negative");
    TORCH_CHECK(valid_len <= max_len, "attn_causal_cache: cur_len + seq_q must not exceed the buffer capacity");
    TORCH_CHECK(
        attention_mass.is_contiguous(),
        "attn_causal_cache: attention_mass must be contiguous (mutated in place)"
    );

    auto qc = q.contiguous();
    auto kc = k_cache.contiguous();
    auto vc = v_cache.contiguous();

    auto o = at::empty_like(qc);

    const int group = static_cast<int>(n_q_heads / n_kv_heads);
    const int64_t blocks = n_q_heads * seq_q;
    const size_t smem = static_cast<size_t>(2 * head_dim + kThreads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, qc.scalar_type(), "attn_causal_cache_cuda", [&] {
        attn_causal_cache_kernel<scalar_t><<<blocks, kThreads, smem, stream>>>(
            qc.data_ptr<scalar_t>(),
            kc.data_ptr<scalar_t>(),
            vc.data_ptr<scalar_t>(),
            attention_mass.data_ptr<float>(),
            o.data_ptr<scalar_t>(),
            static_cast<int>(seq_q),
            max_len,
            valid_len,
            static_cast<int>(head_dim),
            group,
            static_cast<float>(scale),
            cur_len
        );
    });
    return o;
}

}  // namespace pulsar
