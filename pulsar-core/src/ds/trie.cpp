#include "pulsar/ds/trie.hpp"

#include <algorithm>
#include <bit>
#include <memory>
#include <span>
#include <utility>

namespace pulsar::ds {

bool trie<std::uint64_t>::leaf::contains(std::uint8_t index) const {
    return mask & (std::uint64_t{1} << index);
}

void* trie<std::uint64_t>::branch::operator new(std::size_t size, std::size_t count) {
    return ::operator new(size + count * sizeof(node));
}

void trie<std::uint64_t>::branch::operator delete(void* p) {
    ::operator delete(p);
}

std::size_t trie<std::uint64_t>::branch::size() const {
    return static_cast<std::size_t>(std::popcount(mask));
}

auto trie<std::uint64_t>::branch::data() -> node* {
    return reinterpret_cast<node*>(this + 1);
}

auto trie<std::uint64_t>::branch::data() const -> const node* {
    return reinterpret_cast<const node*>(this + 1);
}

auto trie<std::uint64_t>::branch::begin() const -> const node* {
    return data();
}

auto trie<std::uint64_t>::branch::end() const -> const node* {
    return data() + size();
}

auto trie<std::uint64_t>::branch::find(std::uint8_t index) const -> const node* {
    const std::uint64_t p = std::uint64_t{1} << index;

    if (mask & p) {
        return begin() + std::popcount(mask & (p - 1));
    }
    return end();
}

trie<std::uint64_t>::branch::~branch() noexcept {
    if (depth == leaf_depth - 1) {
        return;
    }
    for (node& child : std::span(data(), size())) {
        std::destroy_at(&child.b);
    }
}

trie<std::uint64_t>::node::node(intrusive_ptr<const branch>&& b) noexcept
: b(std::move(b)) {}

trie<std::uint64_t>::node::node(leaf l) noexcept
: l(l) {}

trie<std::uint64_t>::iterator::iterator(
    const std::vector<trie::root>::const_iterator& root,
    const std::vector<trie::root>::const_iterator& roots_end
) noexcept
: root(root),
  roots_end(roots_end) {
    if (root != roots_end) {
        descend(0);
    }
}

void trie<std::uint64_t>::iterator::descend(unsigned int from_depth) noexcept {
    for (unsigned int d = from_depth; d < max_depth; d++) {
        update_state(d);
    }
}

void trie<std::uint64_t>::iterator::update_state(unsigned int depth) noexcept {
    if (depth == leaf_depth) {
        rest[depth] = it[depth - 1]->l.mask;
        return;
    }

    const branch& parent = depth == 0 ? *root->b : *it[depth - 1]->b;
    it[depth] = parent.begin();
    rest[depth] = parent.mask;
}

auto trie<std::uint64_t>::iterator::operator++() noexcept -> iterator& {
    rest[leaf_depth] &= rest[leaf_depth] - 1;
    if (rest[leaf_depth] != 0) {
        return *this;
    }
    for (unsigned int d = leaf_depth; d > 0; d--) {
        ++it[d - 1];
        rest[d - 1] &= rest[d - 1] - 1;
        if (rest[d - 1]) {
            descend(d);
            return *this;
        }
    }
    if (++root != roots_end) {
        descend(0);
    }
    return *this;
}

auto trie<std::uint64_t>::iterator::operator++(int) noexcept -> iterator {
    iterator previous = *this;
    ++*this;
    return previous;
}

std::uint64_t trie<std::uint64_t>::iterator::operator*() const noexcept {
    std::uint64_t key = root->prefix;
    for (unsigned int d = 0; d < max_depth; d++) {
        key = key << offset_width | std::countr_zero(rest[d]);
    }
    return key;
}

bool trie<std::uint64_t>::contains(std::uint64_t key) const noexcept {
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

bool operator==(const trie<std::uint64_t>::iterator& a, const trie<std::uint64_t>::iterator& b) noexcept {
    return a.root == b.root && (a.root == a.roots_end || a.rest == b.rest);
}

trie<std::uint64_t>::trie(const trie& other) noexcept
: roots(other.roots) {}

trie<std::uint64_t>::trie(trie&& other) noexcept
: roots(std::move(other.roots)) {}

trie<std::uint64_t>::iterator trie<std::uint64_t>::begin() const noexcept {
    return trie<std::uint64_t>::iterator(roots.begin(), roots.end());
}

trie<std::uint64_t>::iterator trie<std::uint64_t>::end() const noexcept {
    return trie<std::uint64_t>::iterator(roots.end(), roots.end());
}

}  // namespace pulsar::ds
