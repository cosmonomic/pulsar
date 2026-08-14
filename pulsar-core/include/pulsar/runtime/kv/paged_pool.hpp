#pragma once

#include <ATen/core/Tensor.h>

#include <c10/core/Device.h>

#include <cstdint>
#include <vector>

// Paged K/V storage, shared by every component that holds whole address-aligned
// pages outside the ActiveBuffer: the RecallBuffer (device, relevance-layer K only)
// and PagedBucket (host, full KV). Pages are light descriptors; K/V lives once per
// component in its pool and is never cloned per page.
//
//   slabs_k[s] / slabs_v[s] : [slab_pages, page_size, n_layers, n_kv_heads, head_dim]
//
// SLABBED: the pool is a list of fixed-size slabs, and growth APPENDS one. A slab is
// a whole number of pages, so a page never straddles a boundary, and nothing already
// in the pool is copied or reallocated -- growing by n pages moves 0 bytes and a live
// pool view stays valid across it. A preallocated pool is exactly one slab.
//
// PAGE-MAJOR: a page's K across every layer is one contiguous block, so a whole page
// crosses the PCIe bus in a single direct transfer with no host-side gather. The
// ActiveBuffer's pool is layer-major, so an evict is strided on the DEVICE side only,
// where one batched gather serves the whole cycle.
//
// slabs_v is empty when store_v is false.

namespace pulsar {

// A whole page moving between components: identity + token_ids + its K/V + the
// eviction-time mass snapshot. token_ids is always CPU. mass/mass_step carry the
// page's rank across a spill so the receiving bucket keeps ordering it by the same
// lazy-decayed effective mass. addr (= page_id*page_size + offset) and pos (the
// session's short_offset, the position demotion roped K to) are derived on promotion,
// not carried here.
//
// k/v are one CONTIGUOUS all-layer block each, the page-major pool layout, so a
// bucket hands them back without restaging: a paged bucket returns a narrow() VIEW
// into its pool and a file bucket returns its pread destination, and either uploads
// in one transfer. A returned view stays valid for the rest of the cycle -- slot
// frees are deferred to end_cycle and a slab is never reallocated -- which is exactly
// as long as a promote or a spill needs it.
struct PageKV {
    int64_t page_id = 0;
    at::Tensor token_ids;  // int64 [cnt] CPU
    at::Tensor k;  // [cnt, n_layers, n_kv_heads, head_dim]
    at::Tensor v;  // [cnt, n_layers, n_kv_heads, head_dim]
    double mass = 0.0;
    int64_t mass_step = 0;
};

// One page held in a pool: its permanent page index, its slot block, its token ids
// in RAM, and the eviction-time mass snapshot the spill ranks by.
struct PoolPage {
    int64_t page_id = 0;
    int64_t pool_page = 0;  // slot block base = pool_page * page_size
    std::vector<int64_t> token_ids;
    double mass = 0.0;
    int64_t mass_step = 0;

    int64_t cnt() const {
        return static_cast<int64_t>(this->token_ids.size());
    }
};

// The host-side half of an accept: the pages whose descriptors are already
// inserted and whose destination space is already claimed, paired with the
// ActiveBuffer slot block each page's K/V is copied out of. fill() is the only part
// that touches KV. dst_bases is whatever the receiving medium addresses its
// destination by -- a pool page for a paged bucket, a byte offset for a file
// bucket -- and is only ever read by the bucket that produced it.
struct Reservation {
    std::vector<int64_t> page_ids;
    std::vector<int64_t> src_bases;  // active-pool slot block per page
    std::vector<int64_t> dst_bases;  // receiving bucket's destination per page
};

// A batched gather of pool pages, grouped by the slab that holds them. Each group is
// one index_select landing straight in its own contiguous block of the output, so a
// gather crossing slabs still moves every byte exactly once. Built once for a page
// list and reused for every layer of it.
//
// The groups are in slot order, so the gathered rows are NOT in the requested order:
// `order` maps a gathered row to the position it was asked for.
struct SlabGather {
    struct Group {
        int64_t slab;
        int64_t base;  // the group's first gathered row
        at::Tensor rows;  // int64 [count] in-slab rows, on the pool's device
    };
    std::vector<Group> groups;
    std::vector<int64_t> order;
    int64_t n_pages = 0;
};

struct PagedPool {
    // max_pages > 0 bounds the pool: alloc_page appends slabs up to it and then
    // throws. 0 is unbounded. The bound is rounded UP to a whole slab, so the pool
    // holds at least it.
    // preallocate takes the whole bound at construction as ONE slab, which is what
    // layer_k needs; otherwise the slabs arrive as pages are claimed, so a bound costs
    // only what it is used for.
    // slab_pages fixes the growth slab; 0 derives it from a byte budget, which is what
    // bounds the cost of one growth.
    // pinned page-locks a HOST pool, which is what lets a transfer to or from it be a
    // direct DMA instead of a driver-staged copy through a bounce buffer; ignored on a
    // device pool.
    PagedPool(
        int64_t n_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        at::ScalarType dtype,
        at::Device device,
        bool store_v,
        int64_t max_pages,
        bool pinned = false,
        bool preallocate = false,
        int64_t slab_pages = 0
    );

    // One page's K/V across every layer, [page_size, n_layers, n_kv_heads, head_dim]
    // and contiguous: the unit a whole-page transfer moves.
    at::Tensor page_k(int64_t pool_page) const;
    at::Tensor page_v(int64_t pool_page) const;
    // One layer's pages, [capacity, page_size, n_kv_heads, head_dim]. A strided view,
    // so an index_select over it gathers whole pages of that layer. Needs the whole
    // pool in one slab, so only a preallocated pool has one.
    at::Tensor layer_k(int64_t layer) const;

    // Group the given pool pages by slab; see SlabGather.
    SlabGather plan_gather(const std::vector<int64_t>& pool_pages) const;
    // The planned pages' rows of one layer, [n_pages, page_size, n_kv_heads,
    // head_dim], in the plan's order.
    at::Tensor gather_layer_k(int64_t layer, const SlabGather& plan) const;

    // Append one slab and free its slots.
    void add_slab();
    // Pop a free slot id (LIFO: the most-recently-freed slot, cache-warm). Appends a
    // slab when the free list is empty; a pool already at its bound throws instead.
    int64_t alloc_page();
    void free_page(int64_t pool_page) {
        this->free_pages.push_back(pool_page);
    }

    int64_t n_layers;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t page_size;
    at::ScalarType dtype;
    at::Device device;
    bool store_v;
    bool pinned;
    int64_t slab_pages;
    int64_t max_pages;  // 0 = unbounded; else a whole number of slabs
    std::vector<at::Tensor> slabs_k;  // each [slab_pages, page_size, n_layers, ...]
    std::vector<at::Tensor> slabs_v;  // same shape; empty when !store_v
    int64_t capacity = 0;  // pages the pool holds (>= its live pages)
    std::vector<int64_t> free_pages;  // free slot ids, LIFO stack
};

// A bucket's batched candidate gather: one [n_pages, page_size, n_kv_heads, head_dim]
// tensor per requested layer, in `layers` order. A bucket gathers in ITS OWN storage
// order so that nothing is restaged, so the rows are NOT in the requested page_ids
// order: row i holds page_ids[order[i]]. Every consumer must reindex through `order`.
struct GatheredK {
    std::vector<at::Tensor> layers;
    std::vector<int64_t> order;
};

}  // namespace pulsar
