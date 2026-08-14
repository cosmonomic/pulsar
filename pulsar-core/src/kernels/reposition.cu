#include "pulsar/ops.hpp"
#include "rope_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

// Reposition already-RoPE'd keys in a paged KV pool to new pos.
//
// The pool stores ROPED keys (the driver applies RoPE before write_kv). RoPE is a
// rotation and rotations compose, so moving a key from p_old to p_new is one extra
// rotation by angle (p_new - p_old) * inv_freq[j] applied to the roped key:
//   reposition(rope(raw, p_old), p_old -> p_new) == rope(raw, p_new).
//
// Same "rotate_half" convention and same fp64-angle / fp32-rotation split as
// rope.cu (both call rope_sincos over the same quadrants-per-position table); both
// are required for the two to compose exactly in fp32 (modulo the pool round-trip
// through its stored dtype).
//
// RoPE applies to K only; there is no V variant.
//
// k_pool [num_pages, page_size, n_kv_heads, head_dim] is ONE layer's pool, mutated
// in place. slots/old_positions/new_positions are int32 [M]: for each i, the key at
// physical slot slots[i] (indexing the flattened [num_pages*page_size] first two
// dims) is currently roped at old_positions[i] and is rotated in place to
// new_positions[i]. One block per (i, kv_head); threads stride over the h frequency
// pairs.

namespace pulsar {
namespace {

template <typename scalar_t>
__global__ void reposition_kv_kernel(
    scalar_t* __restrict__ k_pool,
    const int32_t* __restrict__ slots,
    const int32_t* __restrict__ old_positions,
    const int32_t* __restrict__ new_positions,
    const double* __restrict__ quadrants_per_position,
    int64_t n_kv_heads,
    int64_t head_dim
) {
    const int64_t prog = blockIdx.x;
    const int64_t i = prog / n_kv_heads;
    const int64_t head = prog % n_kv_heads;
    const int64_t h = head_dim / 2;

    const double delta = static_cast<double>(new_positions[i]) - static_cast<double>(old_positions[i]);
    if (delta == 0.0) {
        return;  // no-op move
    }

    const int64_t slot = static_cast<int64_t>(slots[i]);
    scalar_t* row = k_pool + (slot * n_kv_heads + head) * head_dim;

    for (int64_t j = threadIdx.x; j < h; j += blockDim.x) {
        float c, sn;
        rope_sincos(delta * quadrants_per_position[j], c, sn);
        const float lo = static_cast<float>(row[j]);
        const float hi = static_cast<float>(row[j + h]);
        row[j] = static_cast<scalar_t>(lo * c - hi * sn);
        row[j + h] = static_cast<scalar_t>(hi * c + lo * sn);
    }
}

}  // namespace

void reposition_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& slots,
    const at::Tensor& old_positions,
    const at::Tensor& new_positions,
    double theta
) {
    TORCH_CHECK(
        k_pool.is_cuda() && slots.is_cuda() && old_positions.is_cuda() && new_positions.is_cuda(),
        "reposition_kv: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        k_pool.dim() == 4,
        "reposition_kv: k_pool must be 4-D [num_pages, page_size, "
        "n_kv_heads, head_dim]"
    );
    TORCH_CHECK(k_pool.is_contiguous(), "reposition_kv: k_pool must be contiguous (mutated in place)");
    TORCH_CHECK(
        slots.scalar_type() == at::kInt && old_positions.scalar_type() == at::kInt &&
            new_positions.scalar_type() == at::kInt,
        "reposition_kv: slots/old_positions/new_positions must be int32"
    );
    TORCH_CHECK(
        slots.dim() == 1 && old_positions.dim() == 1 && new_positions.dim() == 1,
        "reposition_kv: slots/old_positions/new_positions must be 1-D [M]"
    );
    const int64_t M = slots.size(0);
    TORCH_CHECK(
        old_positions.size(0) == M && new_positions.size(0) == M,
        "reposition_kv: slots/old_positions/new_positions must share length M"
    );

    const int64_t num_pages = k_pool.size(0);
    const int64_t page_size = k_pool.size(1);
    const int64_t n_kv_heads = k_pool.size(2);
    const int64_t head_dim = k_pool.size(3);
    TORCH_CHECK(head_dim % 2 == 0, "reposition_kv: head_dim must be even");

    if (M == 0) {
        return;
    }

    auto sl = slots.contiguous();
    auto op = old_positions.contiguous();
    auto np = new_positions.contiguous();
    auto k_flat = k_pool.view({num_pages * page_size, n_kv_heads, head_dim});
    const auto& quadrants_per_position = rope_quadrants_per_position(theta, head_dim, k_pool.device());

    const int64_t blocks = M * n_kv_heads;
    const int threads = static_cast<int>(std::min<int64_t>(256, head_dim / 2));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, k_pool.scalar_type(), "reposition_kv_cuda", [&] {
        reposition_kv_kernel<scalar_t><<<static_cast<int>(blocks), threads, 0, stream>>>(
            k_flat.data_ptr<scalar_t>(),
            sl.data_ptr<int32_t>(),
            op.data_ptr<int32_t>(),
            np.data_ptr<int32_t>(),
            quadrants_per_position.data_ptr<double>(),
            n_kv_heads,
            head_dim
        );
    });
}

}  // namespace pulsar
