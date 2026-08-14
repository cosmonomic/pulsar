#pragma once

#include <ATen/ATen.h>

#include <cmath>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

// Host-side RoPE angle arithmetic, shared by the CUDA kernels (rope.cu and
// reposition.cu, through rope_common.cuh) and by the ATen rotation the recall
// scorer applies (recall.cpp). Every rotation in the engine indexes the same
// frequency table, so the scorer's rotation and the one its keys were stored with
// cannot drift apart.

namespace pulsar {

// Device fp64 table of head_dim/2 RoPE inverse frequencies for a base theta,
// carried in QUADRANTS per unit of position: entry j is
// theta^(-2j/head_dim) * 2/pi, so multiplying by a position yields the angle in
// units of pi/2 and rope_sincos reduces it with one subtraction.
//
// The table MUST stay fp64: its relative error scales by the position into the
// angle, so an fp32 table puts 2e-3 radians on a 32k-position angle.
//
// Cached per (theta, head_dim, device) and never freed; the reference stays valid
// for the process.
inline const at::Tensor& rope_quadrants_per_position(double theta, int64_t head_dim, at::Device device) {
    using Key = std::tuple<double, int64_t, int8_t>;
    static std::mutex guard;
    static auto* cache = new std::map<Key, at::Tensor>();

    const Key key{theta, head_dim, device.index()};
    const std::lock_guard<std::mutex> lock(guard);
    if (const auto found = cache->find(key); found != cache->end()) {
        return found->second;
    }

    constexpr long double two_over_pi = 0.63661977236758134307553505349005745L;
    const int64_t pairs = head_dim / 2;
    std::vector<double> quadrants_per_position(pairs);
    for (int64_t j = 0; j < pairs; ++j) {
        const long double exponent = -2.0L * static_cast<long double>(j) / static_cast<long double>(head_dim);
        quadrants_per_position[j] = static_cast<double>(
            std::pow(static_cast<long double>(theta), exponent) * two_over_pi
        );
    }
    auto table = at::tensor(quadrants_per_position, at::dtype(at::kDouble)).to(device);
    return cache->emplace(key, std::move(table)).first->second;
}

// Rotate x's trailing head_dim axis by `position` TOKENS, in the rotate_half
// convention rope_cuda uses, and to the accuracy rope_cuda reaches: the angle is
// formed in fp64 off the shared table and only cos/sin are rounded to fp32. An
// fp32 angle would instead scale its rounding by the position, reaching 2e-3
// radians at position 32k.
//
// position must broadcast against x's leading axes once given a trailing frequency
// axis, so a 0-dim tensor turns every row by the same angle and a [rows, 1] tensor
// turns each row by its own. RoPE rotations compose, so rotating a key stored at p
// by a delta lands it at p + delta.
inline at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, double theta) {
    constexpr double quadrant_radians = 1.57079632679489661923;
    const int64_t head_dim = x.size(-1), half = head_dim / 2;
    at::Tensor angle = position.to(at::kDouble).unsqueeze(-1) *
        rope_quadrants_per_position(theta, head_dim, x.device()) * quadrant_radians;
    at::Tensor cs = at::cos(angle).to(at::kFloat);
    at::Tensor sn = at::sin(angle).to(at::kFloat);
    at::Tensor first = x.slice(-1, 0, half), second = x.slice(-1, half, head_dim);
    return at::cat({first * cs - second * sn, second * cs + first * sn}, -1);
}

}  // namespace pulsar
