#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <flat_set>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "pulsar/ds/intrusive_ptr.hpp"

namespace pulsar::ds {

class sparse_bitset;

template <typename t_range>
concept key_range = std::ranges::forward_range<t_range> &&
    std::convertible_to<std::ranges::range_reference_t<t_range>, std::uint64_t> &&
    !std::same_as<std::remove_cvref_t<t_range>, sparse_bitset>;

class sparse_bitset {
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
    static constexpr std::uint64_t path_at(std::uint64_t key, std::uint8_t depth) noexcept;

    explicit sparse_bitset(std::vector<root>&& roots) noexcept;

    template <key_range t_range, typename t_op>
    sparse_bitset mutate(t_range&& keys, t_op op) const;

    template <unsigned int depth, std::forward_iterator t_it, std::sentinel_for<t_it> t_sent, typename t_op>
    static intrusive_ptr<const branch> mutate(const branch* existing, t_it& it, const t_sent& last, t_op op);

  public:
    struct iterator;

    sparse_bitset() noexcept = default;

    template <key_range t_range> sparse_bitset(std::sorted_unique_t, t_range&& keys);
    template <key_range t_range> sparse_bitset(std::sorted_equivalent_t, t_range&& keys);

    explicit sparse_bitset(const sparse_bitset& other) noexcept;
    explicit sparse_bitset(sparse_bitset&& other) noexcept;

    sparse_bitset& operator=(const sparse_bitset& other) = default;
    sparse_bitset& operator=(sparse_bitset&& other) noexcept = default;

    friend void swap(sparse_bitset& a, sparse_bitset& b) noexcept {
        a.roots.swap(b.roots);
    }

    iterator begin() const noexcept;
    iterator end() const noexcept;

    bool contains(std::uint64_t key) const noexcept;
    iterator find(std::uint64_t key) const noexcept;
    iterator lower_bound(std::uint64_t key) const noexcept;

    template <key_range t_range> sparse_bitset& insert_range(std::sorted_unique_t, t_range&& keys);
    template <key_range t_range> sparse_bitset& insert_range(std::sorted_equivalent_t, t_range&& keys);

    template <std::forward_iterator t_it, std::sentinel_for<t_it> t_sent>
        requires std::convertible_to<std::iter_reference_t<t_it>, std::uint64_t>
    sparse_bitset& insert(std::sorted_unique_t, t_it first, t_sent last);
    template <std::forward_iterator t_it, std::sentinel_for<t_it> t_sent>
        requires std::convertible_to<std::iter_reference_t<t_it>, std::uint64_t>
    sparse_bitset& insert(std::sorted_equivalent_t, t_it first, t_sent last);

    template <key_range t_range> friend sparse_bitset set_union(const sparse_bitset& a, t_range&& keys);
    template <key_range t_range> friend sparse_bitset set_union(t_range&& keys, const sparse_bitset& a);
    template <key_range t_range> friend sparse_bitset set_difference(const sparse_bitset& a, t_range&& keys);
};

struct sparse_bitset::leaf final {
    const std::uint64_t mask;

    bool contains(std::uint8_t index) const;
};

struct sparse_bitset::branch final {
    mutable std::atomic<std::uint32_t> ref{0};
    const std::uint32_t depth;
    const std::uint64_t mask;

    static void* operator new(std::size_t size, std::size_t count);
    static void operator delete(void* p);

    static branch* allocate(std::uint32_t depth, std::uint64_t mask);

    ~branch() noexcept;

    std::size_t size() const;
    node* data();
    const node* data() const;
    const node* begin() const;
    const node* end() const;

    const node* find(std::uint8_t index) const;

    static const branch* find_branch(const branch* b, std::uint8_t index);
    static std::uint64_t find_leaf(const branch* b, std::uint8_t index);
};

union sparse_bitset::node {
    intrusive_ptr<const branch> b;
    leaf l;

    explicit node(intrusive_ptr<const branch>&& b) noexcept;
    explicit node(leaf l) noexcept;
};

struct sparse_bitset::root {
    std::uint64_t prefix;
    intrusive_ptr<const branch> b;
};

struct sparse_bitset::iterator {
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::uint64_t;
    using difference_type = std::ptrdiff_t;

  private:
    friend sparse_bitset;

    std::vector<sparse_bitset::root>::const_iterator root;
    std::vector<sparse_bitset::root>::const_iterator roots_end;
    std::array<const node*, leaf_depth> it{};
    std::array<std::uint64_t, max_depth> rest{};

    iterator(
        const std::vector<sparse_bitset::root>::const_iterator& root,
        const std::vector<sparse_bitset::root>::const_iterator& roots_end
    ) noexcept;

    void descend(unsigned int from_depth) noexcept;
    void update_state(unsigned int depth) noexcept;
    void skip(unsigned int depth) noexcept;
    void seek(std::uint64_t key) noexcept;

  public:
    iterator() noexcept = default;

    iterator& operator++() noexcept;
    iterator operator++(int) noexcept;

    std::uint64_t operator*() const noexcept;

    friend bool operator==(const iterator& a, const iterator& b) noexcept;
};

inline bool sparse_bitset::leaf::contains(std::uint8_t index) const {
    return mask & (std::uint64_t{1} << index);
}

inline void* sparse_bitset::branch::operator new(std::size_t size, std::size_t count) {
    return ::operator new(size + count * sizeof(node));
}

inline void sparse_bitset::branch::operator delete(void* p) {
    ::operator delete(p);
}

inline sparse_bitset::branch* sparse_bitset::branch::allocate(std::uint32_t depth, std::uint64_t mask) {
    return new (static_cast<std::size_t>(std::popcount(mask))) branch{.depth = depth, .mask = mask};
}

inline std::size_t sparse_bitset::branch::size() const {
    return static_cast<std::size_t>(std::popcount(mask));
}

inline sparse_bitset::node* sparse_bitset::branch::data() {
    return reinterpret_cast<node*>(this + 1);
}

inline const sparse_bitset::node* sparse_bitset::branch::data() const {
    return reinterpret_cast<const node*>(this + 1);
}

inline const sparse_bitset::node* sparse_bitset::branch::begin() const {
    return data();
}

inline const sparse_bitset::node* sparse_bitset::branch::end() const {
    return data() + size();
}

inline const sparse_bitset::node* sparse_bitset::branch::find(std::uint8_t index) const {
    const std::uint64_t p = std::uint64_t{1} << index;

    if (mask & p) {
        return begin() + std::popcount(mask & (p - 1));
    }
    return end();
}

inline const sparse_bitset::branch* sparse_bitset::branch::find_branch(const branch* b, std::uint8_t index) {
    if (!b) {
        return nullptr;
    }
    const node* n = b->find(index);
    return n != b->end() ? n->b.get() : nullptr;
}

inline std::uint64_t sparse_bitset::branch::find_leaf(const branch* b, std::uint8_t index) {
    if (!b) {
        return 0;
    }
    const node* n = b->find(index);
    return n != b->end() ? n->l.mask : 0;
}

inline sparse_bitset::branch::~branch() noexcept {
    if (depth == leaf_depth - 1) {
        return;
    }
    for (node& child : std::span(data(), size())) {
        std::destroy_at(&child.b);
    }
}

inline sparse_bitset::node::node(intrusive_ptr<const branch>&& b) noexcept
: b(std::move(b)) {}

inline sparse_bitset::node::node(leaf l) noexcept
: l(l) {}

inline sparse_bitset::iterator::iterator(
    const std::vector<sparse_bitset::root>::const_iterator& root,
    const std::vector<sparse_bitset::root>::const_iterator& roots_end
) noexcept
: root(root),
  roots_end(roots_end) {
    if (root != roots_end) {
        descend(0);
    }
}

inline void sparse_bitset::iterator::descend(unsigned int from_depth) noexcept {
    for (unsigned int d = from_depth; d < max_depth; d++) {
        update_state(d);
    }
}

inline void sparse_bitset::iterator::update_state(unsigned int depth) noexcept {
    if (depth == leaf_depth) {
        rest[depth] = it[depth - 1]->l.mask;
        return;
    }

    const branch& parent = depth == 0 ? *root->b : *it[depth - 1]->b;
    it[depth] = parent.begin();
    rest[depth] = parent.mask;
}

inline void sparse_bitset::iterator::skip(unsigned int depth) noexcept {
    for (unsigned int d = depth; d > 0; d--) {
        ++it[d - 1];
        rest[d - 1] &= rest[d - 1] - 1;
        if (rest[d - 1]) {
            descend(d);
            return;
        }
    }
    if (++root != roots_end) {
        descend(0);
    }
}

inline void sparse_bitset::iterator::seek(std::uint64_t key) noexcept {
    for (unsigned int d = 0; d < max_depth; d++) {
        const std::uint8_t o = offset_at(key, d);
        const std::uint64_t mask = d == leaf_depth ? it[d - 1]->l.mask : (d == 0 ? *root->b : *it[d - 1]->b).mask;
        const std::uint64_t at_or_above = mask & (~std::uint64_t{0} << o);
        if (at_or_above == 0) {
            skip(d);
            return;
        }
        if (d != leaf_depth) {
            const branch& parent = d == 0 ? *root->b : *it[d - 1]->b;
            it[d] = parent.begin() + std::popcount(mask & ~at_or_above);
        }
        rest[d] = at_or_above;
        if (!(at_or_above & (std::uint64_t{1} << o))) {
            descend(d + 1);
            return;
        }
    }
}

inline sparse_bitset::iterator& sparse_bitset::iterator::operator++() noexcept {
    rest[leaf_depth] &= rest[leaf_depth] - 1;
    if (rest[leaf_depth] != 0) {
        return *this;
    }
    skip(leaf_depth);
    return *this;
}

inline sparse_bitset::iterator sparse_bitset::iterator::operator++(int) noexcept {
    iterator previous = *this;
    ++*this;
    return previous;
}

inline std::uint64_t sparse_bitset::iterator::operator*() const noexcept {
    std::uint64_t key = root->prefix;
    for (unsigned int d = 0; d < max_depth; d++) {
        key = key << offset_width | std::countr_zero(rest[d]);
    }
    return key;
}

inline bool sparse_bitset::contains(std::uint64_t key) const noexcept {
    const std::uint64_t prefix = key >> (offset_width * max_depth);

    const auto r = std::ranges::lower_bound(roots, prefix, {}, &root::prefix);
    if (r == roots.end() || r->prefix != prefix) {
        return false;
    }

    const branch* b = r->b.get();
    for (unsigned int d = 0; d < leaf_depth - 1; d++) {
        if (const node* n = b->find(offset_at(key, d)); n != b->end()) {
            b = n->b.get();
            continue;
        }

        return false;
    }

    if (const node* n = b->find(offset_at(key, leaf_depth - 1)); n != b->end()) {
        return n->l.contains(offset_at(key, leaf_depth));
    }

    return false;
}

inline sparse_bitset::iterator sparse_bitset::find(std::uint64_t key) const noexcept {
    const iterator i = lower_bound(key);
    return i != end() && *i == key ? i : end();
}

inline sparse_bitset::iterator sparse_bitset::lower_bound(std::uint64_t key) const noexcept {
    const std::uint64_t prefix = path_at(key, 0);
    const auto r = std::ranges::lower_bound(roots, prefix, {}, &root::prefix);
    iterator i(r, roots.end());
    if (r != roots.end() && r->prefix == prefix) {
        i.seek(key);
    }
    return i;
}

inline bool operator==(const sparse_bitset::iterator& a, const sparse_bitset::iterator& b) noexcept {
    return a.root == b.root && (a.root == a.roots_end || a.rest == b.rest);
}

inline sparse_bitset::sparse_bitset(std::vector<root>&& roots) noexcept
: roots(std::move(roots)) {}

inline sparse_bitset::sparse_bitset(const sparse_bitset& other) noexcept
: roots(other.roots) {}

inline sparse_bitset::sparse_bitset(sparse_bitset&& other) noexcept
: roots(std::move(other.roots)) {}

inline sparse_bitset::iterator sparse_bitset::begin() const noexcept {
    return sparse_bitset::iterator(roots.begin(), roots.end());
}

inline sparse_bitset::iterator sparse_bitset::end() const noexcept {
    return sparse_bitset::iterator(roots.end(), roots.end());
}

constexpr std::uint8_t sparse_bitset::offset_at(std::uint64_t key, std::uint8_t depth) noexcept {
    constexpr std::uint64_t offset_mask = (std::uint64_t{1} << offset_width) - 1;
    return static_cast<std::uint8_t>(key >> (offset_width * (leaf_depth - depth)) & offset_mask);
}

constexpr std::uint64_t sparse_bitset::path_at(std::uint64_t key, std::uint8_t depth) noexcept {
    return key >> (offset_width * (max_depth - depth));
}

template <key_range t_range>
sparse_bitset::sparse_bitset(std::sorted_unique_t, t_range&& keys)
: sparse_bitset(std::sorted_equivalent, std::forward<t_range>(keys)) {}

template <key_range t_range>
sparse_bitset::sparse_bitset(std::sorted_equivalent_t, t_range&& keys) {
    insert_range(std::sorted_equivalent, std::forward<t_range>(keys));
}

template <key_range t_range>
sparse_bitset& sparse_bitset::insert_range(std::sorted_unique_t, t_range&& keys) {
    return insert_range(std::sorted_equivalent, std::forward<t_range>(keys));
}

template <key_range t_range>
sparse_bitset& sparse_bitset::insert_range(std::sorted_equivalent_t, t_range&& keys) {
    sparse_bitset mutated = set_union(*this, std::forward<t_range>(keys));
    std::ranges::swap(*this, mutated);
    return *this;
}

template <std::forward_iterator t_it, std::sentinel_for<t_it> t_sent>
    requires std::convertible_to<std::iter_reference_t<t_it>, std::uint64_t>
sparse_bitset& sparse_bitset::insert(std::sorted_unique_t, t_it first, t_sent last) {
    return insert_range(std::sorted_equivalent, std::ranges::subrange(std::move(first), std::move(last)));
}

template <std::forward_iterator t_it, std::sentinel_for<t_it> t_sent>
    requires std::convertible_to<std::iter_reference_t<t_it>, std::uint64_t>
sparse_bitset& sparse_bitset::insert(std::sorted_equivalent_t, t_it first, t_sent last) {
    return insert_range(std::sorted_equivalent, std::ranges::subrange(std::move(first), std::move(last)));
}

template <key_range t_range>
sparse_bitset set_union(const sparse_bitset& a, t_range&& keys) {
    return a.mutate(std::forward<t_range>(keys), [](std::uint64_t old, std::uint64_t bits) { return old | bits; });
}

template <key_range t_range>
sparse_bitset set_union(t_range&& keys, const sparse_bitset& a) {
    return set_union(a, std::forward<t_range>(keys));
}

template <key_range t_range>
sparse_bitset set_difference(const sparse_bitset& a, t_range&& keys) {
    return a.mutate(std::forward<t_range>(keys), [](std::uint64_t old, std::uint64_t bits) { return old & ~bits; });
}

template <key_range t_range, typename t_op>
sparse_bitset sparse_bitset::mutate(t_range&& keys, t_op op) const {
    std::vector<root> result;
    result.reserve(roots.size());

    auto r = roots.begin();
    auto it = std::ranges::begin(keys);
    const auto last = std::ranges::end(keys);
    while (it != last) {
        const std::uint64_t prefix = path_at(*it, 0);
        for (; r != roots.end() && r->prefix < prefix; ++r) {
            result.push_back(*r);
        }

        const bool had = r != roots.end() && r->prefix == prefix;
        if (intrusive_ptr<const branch> next = mutate<0>(had ? r->b.get() : nullptr, it, last, op)) {
            result.push_back(root{prefix, std::move(next)});
        }
        if (had) {
            ++r;
        }
    }
    result.insert(result.end(), r, roots.end());
    return sparse_bitset(std::move(result));
}

template <unsigned int depth, std::forward_iterator t_it, std::sentinel_for<t_it> t_sent, typename t_op>
intrusive_ptr<const sparse_bitset::branch>
sparse_bitset::mutate(const branch* existing, t_it& it, const t_sent& last, t_op op) {
    constexpr bool has_leafs = depth == leaf_depth - 1;
    using child = std::conditional_t<has_leafs, std::uint64_t, intrusive_ptr<const branch>>;

    std::array<child, std::size_t{1} << offset_width> changed;
    std::uint64_t changed_mask = 0;

    const std::uint64_t path = path_at(*it, depth);
    while (it != last && path_at(*it, depth) == path) {
        const std::uint8_t o = offset_at(*it, depth);

        if constexpr (has_leafs) {
            const std::uint64_t leaf_path = path_at(*it, depth + 1);
            std::uint64_t bits = 0;
            for (; it != last && path_at(*it, depth + 1) == leaf_path; ++it) {
                bits |= std::uint64_t{1} << offset_at(*it, leaf_depth);
            }
            const std::uint64_t old_mask = branch::find_leaf(existing, o);
            if (const std::uint64_t next = op(old_mask, bits); next != old_mask) {
                changed[o] = next;
                changed_mask |= std::uint64_t{1} << o;
            }
        } else {
            const branch* old_branch = branch::find_branch(existing, o);
            if (intrusive_ptr<const branch> next = mutate<depth + 1>(old_branch, it, last, op);
                next.get() != old_branch) {
                changed[o] = std::move(next);
                changed_mask |= std::uint64_t{1} << o;
            }
        }
    }

    if (changed_mask == 0) {
        return intrusive_ptr<const branch>(existing);
    }

    std::uint64_t mask = (existing ? existing->mask : 0) | changed_mask;
    for (std::uint64_t rest = changed_mask; rest != 0; rest &= rest - 1) {
        const unsigned int o = std::countr_zero(rest);
        if (!changed[o]) {
            mask &= ~(std::uint64_t{1} << o);
        }
    }
    if (mask == 0) {
        return {};
    }

    branch* b = branch::allocate(depth, mask);
    node* out = b->data();
    for (std::uint64_t rest = mask; rest != 0; rest &= rest - 1) {
        const unsigned int o = std::countr_zero(rest);
        const bool is_changed = changed_mask & (std::uint64_t{1} << o);
        if constexpr (has_leafs) {
            std::construct_at(out++, leaf{is_changed ? changed[o] : branch::find_leaf(existing, o)});
        } else {
            std::construct_at(
                out++,
                is_changed ? std::move(changed[o]) : intrusive_ptr<const branch>(branch::find_branch(existing, o))
            );
        }
    }
    return intrusive_ptr<const branch>(b);
}

}  // namespace pulsar::ds
