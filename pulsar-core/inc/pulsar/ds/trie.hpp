#pragma once

#include <array>
#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <ranges>
#include <type_traits>
#include <vector>

#include "pulsar/ds/intrusive_ptr.hpp"

namespace pulsar::ds {

template <typename t_key, typename t_value = void> class trie;

template <typename t_range, typename t_key>
concept key_range = std::ranges::forward_range<t_range> &&
    std::convertible_to<std::ranges::range_reference_t<t_range>, t_key> &&
    !std::same_as<std::remove_cvref_t<t_range>, trie<t_key>>;

template <> class trie<std::uint64_t> {
    static constexpr unsigned int offset_width = 6;
    static constexpr unsigned int max_depth = 6;

    static constexpr unsigned int leaf_depth = max_depth - 1;
    static constexpr unsigned int prefix_width = 64 - offset_width * max_depth;

    union node;
    struct root;
    struct branch;
    struct leaf;

    std::vector<root> roots;

    static constexpr std::uint8_t offset_at(std::uint64_t key, std::uint8_t depth) noexcept;

  public:
    struct iterator;

    trie() noexcept = default;

    template <typename t_container> trie(const t_container& roots) noexcept;

    explicit trie(const trie& other) noexcept;
    explicit trie(trie&& other) noexcept;

    iterator begin() const noexcept;
    iterator end() const noexcept;

    bool contains(std::uint64_t key) const noexcept;

    trie& operator|=(const trie& other);
    template <key_range<std::uint64_t> t_range> trie& operator|=(t_range&& keys);

    trie& operator-=(const trie& other);
    template <key_range<std::uint64_t> t_range> trie& operator-=(t_range&& keys);

    friend trie operator|(const trie& a, const trie& b);
    template <key_range<std::uint64_t> t_range> friend trie operator|(const trie& a, t_range&& b);
    template <key_range<std::uint64_t> t_range> friend trie operator|(t_range&& a, const trie& b);

    friend trie operator-(const trie& a, const trie& b);
    template <key_range<std::uint64_t> t_range> friend trie operator-(const trie& a, t_range&& b);
};

struct trie<std::uint64_t>::leaf final {
    const std::uint64_t mask;

    bool contains(std::uint8_t index) const;
};

struct trie<std::uint64_t>::branch final {
    mutable std::atomic<std::uint32_t> ref{0};
    const std::uint32_t depth;
    const std::uint64_t mask;

    static void* operator new(std::size_t size, std::size_t count);
    static void operator delete(void* p);

    ~branch() noexcept;

    std::size_t size() const;
    node* data();
    const node* data() const;
    const node* begin() const;
    const node* end() const;

    const node* find(std::uint8_t index) const;
};

union trie<std::uint64_t>::node {
    intrusive_ptr<const branch> b;
    leaf l;

    explicit node(intrusive_ptr<const branch>&& b) noexcept;
    explicit node(leaf l) noexcept;
};

struct trie<std::uint64_t>::root {
    std::uint64_t prefix;
    intrusive_ptr<const branch> b;
};

struct trie<std::uint64_t>::iterator {
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::uint64_t;
    using difference_type = std::ptrdiff_t;

  private:
    friend trie;

    std::vector<trie::root>::const_iterator root;
    std::vector<trie::root>::const_iterator roots_end;
    std::array<const node*, leaf_depth> it;
    std::array<std::uint64_t, max_depth> rest;

    iterator(
        const std::vector<trie::root>::const_iterator& root,
        const std::vector<trie::root>::const_iterator& roots_end
    ) noexcept;

    void descend(unsigned int from_depth) noexcept;
    void update_state(unsigned int depth) noexcept;

  public:
    iterator& operator++() noexcept;
    iterator operator++(int) noexcept;

    std::uint64_t operator*() const noexcept;

    friend bool operator==(const iterator& a, const iterator& b) noexcept;
};

template <typename t_container>
inline trie<std::uint64_t>::trie(const t_container& roots) noexcept
: roots(std::from_range, roots) {}

constexpr std::uint8_t trie<std::uint64_t>::offset_at(std::uint64_t key, std::uint8_t depth) noexcept {
    constexpr std::uint64_t offset_mask = (std::uint64_t{1} << offset_width) - 1;
    return static_cast<std::uint8_t>(key >> (offset_width * (leaf_depth - depth)) & offset_mask);
}

}  // namespace pulsar::ds
