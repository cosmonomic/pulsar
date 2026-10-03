#pragma once

// Draft interface.

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "pulsar/ds/sparse_bitset.hpp"

namespace pulsar {

class kv_cache {
  public:
    using page_id = std::uint64_t;
    using step = std::uint64_t;

    struct write_run {
        page_id page;
        std::uint32_t offset;
        std::uint32_t count;
    };

    struct batch_tables {
        at::Tensor page_tables;  // int32 [num_views, max_pages]
        at::Tensor cu_view_pages;  // int32 [num_views + 1]
    };

  private:
    struct lane;

    at::Tensor k;  // [n_layers, n_lanes, page_size, n_kv_heads, head_dim]
    at::Tensor v;
    std::vector<lane> lanes;
    std::unordered_map<page_id, std::uint32_t> lane_of;

  public:
    kv_cache(
        std::size_t n_layers,
        std::size_t n_lanes,
        std::size_t page_size,
        std::size_t n_kv_heads,
        std::size_t head_dim,
        at::ScalarType dtype,
        at::Device device
    );

    step begin_step();
    void end_step();

    bool resident(page_id page) const noexcept;

    // Assigns a lane to each page, reusing least recently used lanes not touched by an in-flight step. Returns the
    // pages that lost their lane.
    std::vector<page_id> claim(std::span<const page_id> pages);
    void copy(page_id src, page_id dst);
    // k, v: [n_layers, page_size, n_kv_heads, head_dim]; the step that reads the page waits for the copy.
    void load(page_id page, const at::Tensor& k, const at::Tensor& v);
    void drop(std::span<const page_id> pages) noexcept;

    // Marks every referenced page as used by the current step.
    batch_tables tables(std::span<const ds::sparse_bitset* const> views);
    at::Tensor slots(std::span<const write_run> runs) const;

    at::Tensor k_pool(std::size_t layer) const;
    at::Tensor v_pool(std::size_t layer) const;
};

struct kv_cache::lane {
    page_id page;
    step last_used;
};

}  // namespace pulsar
