#pragma once

#include <c10/util/Exception.h>

#include <cuda_runtime.h>

#include <type_traits>

// Launch geometry, block reduction and template-instantiation ladders shared by
// the attention kernels (attention.cu, paged_attention.cu, paged_prefill.cu).

namespace pulsar {
namespace attn {

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
