#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>
#include <variant>
#include <vector>

#include "pulsar/ds/intrusive_ptr.hpp"

namespace pulsar::ds {

template <typename t_key, typename t_value = void> class trie;

template <> class trie<std::uint64_t> {
    static constexpr int level_bits = 6;
    static constexpr int depth = 6;
    static constexpr int leaf_level = depth - 1;
    static constexpr int root_shift = level_bits * depth;

    struct branch;

    struct leaf final {
        mutable std::atomic<std::size_t> ref{0};
        std::uint64_t mask;
    };

    using node = std::variant<intrusive_ptr<branch>, intrusive_ptr<leaf>>;

    struct branch final {
        mutable std::atomic<std::size_t> ref{0};
        std::uint64_t mask;

        static void* operator new(std::size_t size, std::size_t count);
        static void operator delete(void* p);

        std::size_t size() const;
        node* data();
        const node* data() const;
        node* begin();
        const node* begin() const;
        node* end();
        const node* end() const;

        ~branch();
    };

    static_assert(alignof(node) <= alignof(branch));

    std::vector<intrusive_ptr<branch>> nodes;

  public:
    class iterator {
        struct frame {
            const branch* n;
            std::uint64_t pending;

            std::size_t child_index() const;
        };

        std::span<const intrusive_ptr<branch>> roots;
        std::size_t root = 0;
        std::array<frame, depth> path{};

        bool fill(int level);
        void seek_root(std::size_t from);

      public:
        using iterator_concept = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::uint64_t;
        using difference_type = std::ptrdiff_t;

        iterator() noexcept = default;
        iterator(std::span<const intrusive_ptr<branch>> roots, std::size_t from) noexcept;

        std::uint64_t operator*() const noexcept;
        iterator& operator++() noexcept;
        iterator operator++(int) noexcept;

        friend bool operator==(const iterator& a, const iterator& b) noexcept;
    };

    trie() noexcept = default;

    template <typename t_container> trie(const t_container& nodes) noexcept;

    explicit trie(const trie& other) noexcept;
    explicit trie(trie&& other) noexcept;

    iterator begin() const noexcept;
    iterator end() const noexcept;
};

template <typename t_container>
inline trie<std::uint64_t>::trie(const t_container& nodes) noexcept
: nodes(nodes) {}

}  // namespace pulsar::ds
