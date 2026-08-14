#include "pulsar/ops.hpp"
#include "rope_common.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

// Rotary position embedding (RoPE), HF/Qwen2 "rotate_half" convention.
//
// For head_dim = 2h and inv_freq[j] = theta^(-2j/head_dim), each row s at
// pos p = pos[s] is rotated by angle a = p * inv_freq[j]:
//   out[j]   = x[j]   * cos(a) - x[j+h] * sin(a)
//   out[j+h] = x[j+h] * cos(a) + x[j]   * sin(a)
// The full head_dim is rotated (no partial-rotary fraction). The angle is formed
// in fp64 (see rope_sincos), the rotation runs in fp32, and the result is cast
// back to x's dtype on store.
//
// pos is int64 [seq] and may be arbitrary / non-contiguous, which lets
// the same op re-RoPE a cache under a recompacted pos layout. One block
// handles one (head, row) pair; threads stride over the h frequency pairs.

namespace pulsar {
namespace {

template <typename scalar_t>
__global__ void rope_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ pos,
    const double* __restrict__ quadrants_per_position,
    scalar_t* __restrict__ out,
    int64_t seq,
    int64_t head_dim
) {
    const int64_t blk = blockIdx.x;
    const int64_t s = blk % seq;
    const int64_t h = head_dim / 2;

    const scalar_t* xr = x + blk * head_dim;
    scalar_t* outr = out + blk * head_dim;
    const double p = static_cast<double>(pos[s]);

    for (int64_t j = threadIdx.x; j < h; j += blockDim.x) {
        float c, sn;
        rope_sincos(p * quadrants_per_position[j], c, sn);
        const float lo = static_cast<float>(xr[j]);
        const float hi = static_cast<float>(xr[j + h]);
        outr[j] = static_cast<scalar_t>(lo * c - hi * sn);
        outr[j + h] = static_cast<scalar_t>(hi * c + lo * sn);
    }
}

}  // namespace

at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, double theta) {
    TORCH_CHECK(x.is_cuda() && pos.is_cuda(), "rope: inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 3, "rope: x must be 3-D [n_heads, seq, head_dim]");
    TORCH_CHECK(pos.dim() == 1, "rope: pos must be 1-D [seq]");
    TORCH_CHECK(pos.scalar_type() == at::kLong, "rope: pos must be int64");

    const int64_t n_heads = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t head_dim = x.size(2);

    TORCH_CHECK(head_dim % 2 == 0, "rope: head_dim must be even");
    TORCH_CHECK(pos.size(0) == seq, "rope: pos length must match x seq dim");

    auto xc = x.contiguous();
    auto pc = pos.contiguous();
    auto out = at::empty_like(xc);
    const auto& quadrants_per_position = rope_quadrants_per_position(theta, head_dim, xc.device());

    const int64_t blocks = n_heads * seq;
    const int threads = static_cast<int>(std::min<int64_t>(256, head_dim / 2));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rope_cuda", [&] {
        rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            xc.data_ptr<scalar_t>(),
            pc.data_ptr<int64_t>(),
            quadrants_per_position.data_ptr<double>(),
            out.data_ptr<scalar_t>(),
            seq,
            head_dim
        );
    });
    return out;
}

}  // namespace pulsar
