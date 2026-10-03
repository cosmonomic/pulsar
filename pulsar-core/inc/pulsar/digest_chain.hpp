#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsar/ds/sparse_bitset.hpp"

namespace pulsar {

class digest_chain {
  public:
    struct digest {
        std::uint64_t hi;
        std::uint64_t lo;

        friend bool operator==(const digest& a, const digest& b) noexcept = default;
    };

    struct node;

  private:
    struct digest_hash {
        std::size_t operator()(const digest& d) const noexcept {
            return d.lo;
        }
    };

    std::unordered_map<digest, node, digest_hash> nodes;

  public:
    static digest root(std::uint64_t config_hash) noexcept;

    // The digest is hash(parent, tokens). When it is already present, the existing node is returned unchanged.
    std::pair<digest, const node&> seal(
        const digest& parent,
        std::uint64_t chunk_id,
        std::vector<std::int64_t> tokens,
        ds::sparse_bitset view,
        std::vector<float> density
    );
    void release(const digest& d) noexcept;

    const node* find(const digest& d) const noexcept;
    std::size_t size() const noexcept;
};

struct digest_chain::node final {
    digest parent;
    std::uint64_t chunk_id;
    std::vector<std::int64_t> tokens;
    ds::sparse_bitset view;
    // Aligned with the iteration order of view.
    std::vector<float> density;
};

}  // namespace pulsar
