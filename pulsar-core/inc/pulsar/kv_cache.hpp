#pragma once

// Draft interface.

#include <ATen/core/Tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"

namespace pulsar {

struct page {
    at::Tensor k;  // [n_layers, page_size, n_kv_heads, head_dim]
    at::Tensor v;
};

// fetch returns the pages in the order of hashes, nullopt for those it does not have. put takes ownership of host
// pages and their hashes, matched by index, and returns without waiting for the write.
template <typename t>
concept page_store = requires(t s, std::span<const digest> hashes, std::vector<digest> keys, std::vector<page> pages) {
    { s.fetch(hashes) } -> std::same_as<task<std::vector<std::optional<page>>>>;
    { s.put(std::move(keys), std::move(pages)) } -> std::same_as<void>;
};

enum class page_error { miss, oom };

template <page_store t_store> class kv_cache {
  private:
    struct line;
    struct fetch;

    at::Tensor k;  // [n_layers, n_lines, page_size, n_kv_heads, head_dim]
    at::Tensor v;
    // Indexed by line.
    std::vector<line> lines;
    std::unordered_map<digest, std::uint32_t, digest_hash> digest_to_line_id;
    // Pages a get is fetching into a line, not yet in digest_to_line_id.
    std::unordered_map<digest, fetch, digest_hash> fetching;
    t_store store;

  public:
    kv_cache(
        std::size_t n_layers,
        std::size_t n_lines,
        std::size_t page_size,
        std::size_t n_kv_heads,
        std::size_t head_dim,
        at::ScalarType dtype,
        at::Device device,
        t_store&& store
    );

    // Callers release lines only after the GPU work using them completes. get and allocate never wait for lines. A line
    // is free when it is not held; free lines are reused least recently released first and their pages dropped. Without
    // enough free lines the call fails with oom. A failed call holds nothing.

    // Gives every page a line in the order of hashes, fetching the pages that are not resident from the store, and
    // holds the lines until release. A page another get is fetching shares that line and fetch. Fails with miss when
    // the store does not have a page.
    task<std::expected<std::vector<std::uint32_t>, page_error>> get(std::span<const digest> hashes);
    // Gives count empty pages and holds them until release. A page is not cached until mapped: get never finds it and
    // its line is freed when released.
    task<std::expected<std::vector<std::uint32_t>, page_error>> allocate(std::size_t count);
    // Gives count unmapped lines holding a copy of the page at line_id, held until release.
    task<std::expected<std::vector<std::uint32_t>, page_error>> clone(std::uint32_t line_id, std::size_t count);
    // Caches the full page at each line under the hash at the same index, and puts a host copy of each into the store.
    // Completes once the copies are handed to put.
    task<void> map(std::span<const digest> hashes, std::span<const std::uint32_t> line_ids);
    void release(std::span<const std::uint32_t> line_ids) noexcept;

    at::Tensor k_pool(std::size_t layer) const;
    at::Tensor v_pool(std::size_t layer) const;
};

template <page_store t_store> struct kv_cache<t_store>::line {
    std::optional<digest> hash;
    std::uint64_t last_released;
    std::uint32_t ref;
};

// done is set when the fetch completes. The page arrived when its line's hash then equals its digest.
template <page_store t_store> struct kv_cache<t_store>::fetch {
    std::uint32_t line_id;
    event done;
};

}  // namespace pulsar
