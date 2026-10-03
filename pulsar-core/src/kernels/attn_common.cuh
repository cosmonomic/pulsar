#pragma once

#include "pulsar/kernels/attn_params.cuh"

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>

#include <optional>
#include <type_traits>

// The validated batch, launch geometry, block reduction and template-instantiation
// ladders shared by the paged attention ops (paged_attention.cu, paged_prefill.cu,
// paged_scalar.cu).

namespace pulsar {
namespace attn {

// One validated call of a paged attention op. cu_seqlens_q is undefined for decode,
// where sequence s has the one query row s. mass, cu_view_pages and mass_decay are all
// defined or all undefined.
struct PagedAttnBatch {
    at::Tensor q;
    at::Tensor k_pool;
    at::Tensor v_pool;
    at::Tensor page_tables;
    at::Tensor cu_seqlens_q;
    at::Tensor seqlens_k;
    at::Tensor rope_layout;
    at::Tensor rope_turns;
    at::Tensor mass;
    at::Tensor cu_view_pages;
    at::Tensor mass_decay;
    at::Tensor o;
    at::Tensor lse;
    double mass_length_gain;
    double scale;
    int64_t num_seqs;
    int64_t total_q;
    int64_t n_q_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t page_size;
    int64_t max_pages;

    template <typename scalar_t> PagedAttnParams<scalar_t> params() const {
        auto data = [](const at::Tensor& t) { return t.defined() ? t.data_ptr() : nullptr; };
        return PagedAttnParams<scalar_t>{
            static_cast<const scalar_t*>(data(this->q)),
            static_cast<const scalar_t*>(data(this->k_pool)),
            static_cast<const scalar_t*>(data(this->v_pool)),
            static_cast<const int32_t*>(data(this->page_tables)),
            static_cast<const int32_t*>(data(this->cu_seqlens_q)),
            static_cast<const int32_t*>(data(this->seqlens_k)),
            static_cast<const int32_t*>(data(this->rope_layout)),
            static_cast<const uint64_t*>(data(this->rope_turns)),
            static_cast<float*>(data(this->mass)),
            static_cast<const int32_t*>(data(this->cu_view_pages)),
            static_cast<const float*>(data(this->mass_decay)),
            static_cast<float>(this->mass_length_gain),
            static_cast<scalar_t*>(data(this->o)),
            static_cast<float*>(data(this->lse)),
            static_cast<int>(this->num_seqs),
            static_cast<int>(this->n_q_heads),
            static_cast<int>(this->n_kv_heads),
            static_cast<int>(this->head_dim),
            static_cast<int>(this->max_pages),
            static_cast<int>(this->n_q_heads / this->n_kv_heads),
            this->mass.defined() ? static_cast<int>(this->mass.size(0)) : 0,
            static_cast<float>(this->scale),
        };
    }

    // fp16/bf16 with a (page_size, head_dim) the tensor-core kernels are compiled for.
    bool tensor_core_eligible() const;
};

// Checks the shared argument contract (see pulsar/ops.hpp) and allocates o. op names
// the caller in every error.
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
);

// The scalar kernel over the whole batch: the fallback for shapes the tensor-core
// kernels are not compiled for, and the reference both are checked against.
void launch_attn_scalar(const PagedAttnBatch& batch, cudaStream_t stream);

// Upper bound on head_dim; sizes the scalar kernels' per-thread output
// accumulator, which lives in local memory.
constexpr int kMaxHeadDim = 256;

// CTA width of the scalar kernels. Power of two: block_reduce_* requires it.
constexpr int kThreads = 128;

// Tree reduction of one value per thread over a CTA of NTHREADS threads.
// NTHREADS must be a power of two and every thread of the CTA must call this.
// `scratch` is NTHREADS floats of shared memory, reusable once this returns; the
// result is returned to every thread.
template <int NTHREADS> __device__ __forceinline__ float block_reduce_max(float* scratch, int tid, float value) {
    scratch[tid] = value;
    __syncthreads();
    for (int stride = NTHREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

template <int NTHREADS> __device__ __forceinline__ float block_reduce_sum(float* scratch, int tid, float value) {
    scratch[tid] = value;
    __syncthreads();
    for (int stride = NTHREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Call fn with the tensor-core kernels' (scalar_t, BF16) for a fp16 or bf16 dtype.
template <typename Fn> void dispatch_tensor_core_dtype(at::ScalarType dtype, Fn&& fn) {
    if (dtype == at::kBFloat16) {
        fn(std::type_identity<at::BFloat16>{}, std::true_type{});
    } else {
        fn(std::type_identity<at::Half>{}, std::false_type{});
    }
}

// Call fn with the compile-time page_size tag matching the runtime page_size.
template <typename Fn> void dispatch_page_size(const char* op, int64_t page_size, Fn&& fn) {
    if (page_size == 16) {
        fn(std::integral_constant<int, 16>{});
    } else if (page_size == 32) {
        fn(std::integral_constant<int, 32>{});
    } else {
        TORCH_CHECK(false, op, ": unsupported page_size ", page_size, " (compiled for 16 and 32)");
    }
}

// Call fn with the compile-time (page_size, head_dim) tags matching the runtime
// values. The tensor-core kernels exist only for these pairs, so an unsupported
// one is rejected rather than silently computed under the wrong geometry.
template <typename Fn> void dispatch_page_head(const char* op, int64_t page_size, int64_t head_dim, Fn&& fn) {
    dispatch_page_size(op, page_size, [&](auto page_tag) {
        if (head_dim == 64) {
            fn(page_tag, std::integral_constant<int, 64>{});
        } else if (head_dim == 128) {
            fn(page_tag, std::integral_constant<int, 128>{});
        } else {
            TORCH_CHECK(false, op, ": unsupported head_dim ", head_dim, " (tensor-core path compiled for 64 and 128)");
        }
    });
}

}  // namespace attn
}  // namespace pulsar
