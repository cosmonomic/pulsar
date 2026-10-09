#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace pulsar {

struct digest {
    std::uint64_t hi;
    std::uint64_t lo;

    friend bool operator==(const digest& a, const digest& b) noexcept = default;
};

struct digest_hash {
    std::size_t operator()(const digest& d) const noexcept {
        return d.lo;
    }
};

// The root input is seed; the page input is parent.hi, parent.lo, then tokens, each as 8 little-endian bytes. The
// domain is "pulsar root digest" or "pulsar page digest". SHA-256 hashes domain followed by input; BLAKE3 hashes input
// in derive-key mode with domain as context. The digest is the first 16 output bytes read as hi, then lo, each
// little-endian. Every service computing digests must match this encoding and the hash chosen at build time.
digest make_digest(std::span<const std::byte> seed) noexcept;
digest make_digest(const digest& parent, std::span<const std::int64_t> tokens) noexcept;

}  // namespace pulsar
