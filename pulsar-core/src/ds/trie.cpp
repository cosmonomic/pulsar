#include "pulsar/ds/trie.hpp"

#include <bit>
#include <memory>
#include <utility>

namespace pulsar::ds {

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

auto trie<std::uint64_t>::branch::begin() -> node* {
    return data();
}

auto trie<std::uint64_t>::branch::begin() const -> const node* {
    return data();
}

auto trie<std::uint64_t>::branch::end() -> node* {
    return data() + size();
}

auto trie<std::uint64_t>::branch::end() const -> const node* {
    return data() + size();
}

trie<std::uint64_t>::branch::~branch() {
    std::destroy(begin(), end());
}

std::size_t trie<std::uint64_t>::iterator::frame::child_index() const {
    return static_cast<std::size_t>(std::popcount(n->mask) - std::popcount(pending));
}

bool trie<std::uint64_t>::iterator::fill(int level) {
    while (true) {
        if (path[level].pending == 0) {
            if (level == 0) {
                return false;
            }
            --level;
            path[level].pending &= path[level].pending - 1;
        } else if (level == leaf_level) {
            return true;
        } else {
            const auto& next = path[level].n->data()[path[level].child_index()];
            ++level;
            if (level == leaf_level) {
                path[level] = {nullptr, std::get<intrusive_ptr<leaf>>(next)->mask};
            } else {
                const auto& inner = std::get<intrusive_ptr<branch>>(next);
                path[level] = {&*inner, inner->mask};
            }
        }
    }
}

void trie<std::uint64_t>::iterator::seek_root(std::size_t from) {
    for (root = from; root < roots.size(); ++root) {
        if (roots[root]) {
            path[0] = {&*roots[root], roots[root]->mask};
            if (fill(0)) {
                return;
            }
        }
    }
}

trie<std::uint64_t>::iterator::iterator(std::span<const intrusive_ptr<branch>> roots, std::size_t from) noexcept
: roots(roots) {
    seek_root(from);
}

std::uint64_t trie<std::uint64_t>::iterator::operator*() const noexcept {
    auto value = static_cast<std::uint64_t>(root) << root_shift;
    for (int level = 0; level < depth; ++level) {
        auto offset = static_cast<std::uint64_t>(std::countr_zero(path[level].pending));
        value |= offset << (level_bits * (leaf_level - level));
    }
    return value;
}

auto trie<std::uint64_t>::iterator::operator++() noexcept -> iterator& {
    path[leaf_level].pending &= path[leaf_level].pending - 1;
    if (!fill(leaf_level)) {
        seek_root(root + 1);
    }
    return *this;
}

auto trie<std::uint64_t>::iterator::operator++(int) noexcept -> iterator {
    auto previous = *this;
    ++*this;
    return previous;
}

bool operator==(const trie<std::uint64_t>::iterator& a, const trie<std::uint64_t>::iterator& b) noexcept {
    return a.root == b.root && (a.root == a.roots.size() || *a == *b);
}

trie<std::uint64_t>::trie(const trie& other) noexcept
: trie(other.nodes) {}

trie<std::uint64_t>::trie(trie&& other) noexcept
: trie(std::move(other.nodes)) {}

auto trie<std::uint64_t>::begin() const noexcept -> iterator {
    return iterator(nodes, 0);
}

auto trie<std::uint64_t>::end() const noexcept -> iterator {
    return iterator(nodes, nodes.size());
}

}  // namespace pulsar::ds
