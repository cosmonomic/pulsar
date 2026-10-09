#include "pulsar/digest.hpp"

#include <blake3.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace pulsar {

static_assert(std::endian::native == std::endian::little);

namespace {

class hasher {
    blake3_hasher state;

  public:
    explicit hasher(std::string_view domain) noexcept {
        blake3_hasher_init_derive_key_raw(&state, domain.data(), domain.size());
    }

    void update(const void* data, std::size_t size) noexcept {
        blake3_hasher_update(&state, data, size);
    }

    digest finalize() noexcept {
        std::array<std::uint8_t, 16> out;
        blake3_hasher_finalize(&state, out.data(), out.size());
        digest d;
        std::memcpy(&d.hi, out.data(), sizeof(d.hi));
        std::memcpy(&d.lo, out.data() + sizeof(d.hi), sizeof(d.lo));
        return d;
    }
};

}  // namespace

digest make_digest(std::span<const std::byte> seed) noexcept {
    hasher h("pulsar root digest");
    h.update(seed.data(), seed.size());
    return h.finalize();
}

digest make_digest(const digest& parent, std::span<const std::int64_t> tokens) noexcept {
    hasher h("pulsar page digest");
    h.update(&parent.hi, sizeof(parent.hi));
    h.update(&parent.lo, sizeof(parent.lo));
    h.update(tokens.data(), tokens.size_bytes());
    return h.finalize();
}

}  // namespace pulsar
