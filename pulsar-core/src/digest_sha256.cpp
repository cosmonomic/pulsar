#include "pulsar/digest.hpp"

#include <openssl/evp.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>

namespace pulsar {

static_assert(std::endian::native == std::endian::little);

namespace {

void check(int openssl_result) noexcept {
    if (openssl_result != 1) {
        std::abort();
    }
}

const EVP_MD* sha256() noexcept {
    static const EVP_MD* md = EVP_MD_fetch(nullptr, "SHA256", nullptr);
    if (md == nullptr) {
        std::abort();
    }
    return md;
}

class hasher {
    struct context_deleter {
        void operator()(EVP_MD_CTX* ctx) const noexcept {
            EVP_MD_CTX_free(ctx);
        }
    };

    std::unique_ptr<EVP_MD_CTX, context_deleter> context{EVP_MD_CTX_new()};

  public:
    explicit hasher(std::string_view domain) noexcept {
        if (context == nullptr) {
            std::abort();
        }
        check(EVP_DigestInit_ex2(context.get(), sha256(), nullptr));
        update(domain.data(), domain.size());
    }

    void update(const void* data, std::size_t size) noexcept {
        check(EVP_DigestUpdate(context.get(), data, size));
    }

    digest finalize() noexcept {
        std::array<std::uint8_t, 32> out;
        check(EVP_DigestFinal_ex(context.get(), out.data(), nullptr));
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
