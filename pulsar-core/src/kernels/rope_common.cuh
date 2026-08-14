#pragma once

#include "pulsar/rope.hpp"

// Device-side RoPE angle arithmetic for rope.cu and reposition.cu. Both form the
// same angle -- an absolute position times the inverse frequency for rope, a
// position delta times it for reposition -- so both index the shared table in
// pulsar/rope.hpp.

namespace pulsar {

// cos/sin of a RoPE angle given in QUADRANTS (the angle divided by pi/2), rounded
// to fp32.
//
// The argument arrives in fp64 and the whole-quadrant part is removed there: that
// subtraction is exact, so only the remainder, bounded by half a quadrant, is
// rounded to fp32. Rounding the unreduced angle instead would scale the rounding
// error by the angle's magnitude, which reaches the trained context length.
__device__ __forceinline__ void rope_sincos(double quadrants, float& cos_out, float& sin_out) {
    const double whole = nearbyint(quadrants);
    const float reduced = static_cast<float>(quadrants - whole) * 1.57079632679489662f;

    float sine, cosine;
    sincosf(reduced, &sine, &cosine);
    switch (static_cast<int>(static_cast<long long>(whole)) & 3) {
    case 0:
        cos_out = cosine;
        sin_out = sine;
        break;
    case 1:
        cos_out = -sine;
        sin_out = cosine;
        break;
    case 2:
        cos_out = -cosine;
        sin_out = -sine;
        break;
    default:
        cos_out = sine;
        sin_out = -cosine;
        break;
    }
}

}  // namespace pulsar
