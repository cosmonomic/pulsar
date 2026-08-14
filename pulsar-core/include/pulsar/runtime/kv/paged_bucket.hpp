#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"  // ActiveBuffer, DemotedKV
#include "pulsar/runtime/kv/page_view.hpp"
#include "pulsar/runtime/kv/paged_pool.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

// PagedBucket: a spill bucket backed by a paged pool. One PagedPool per sequence
// holds the demoted pages' full KV (all layers, K and V); page identity (page_id,
// token_ids, mass) stays in a host descriptor map with a slot free list.
// Device-parameterized: the engine's S1 is a host-RAM instance, so eviction is the
// one D2H the cycle needs and everything downstream reads host memory. A host pool is
// PAGE-LOCKED, so that D2H and the promote's H2D are direct DMAs into and out of the
// pool itself.
//
// Slot frees are DEFERRED: take_pages and spill_to_budget mark a slot dead and
// release_pending() returns it to the free list at end-of-cycle, so a page's K/V
// stays readable for the whole cycle that consumed it.

namespace pulsar {

struct BucketPool;  // one sequence's pool + descriptor map; defined in paged_bucket.cpp
struct FillStream;  // the side stream + its fences; defined in paged_bucket.cpp

struct PagedBucket {
    // budget_pages > 0 bounds each sequence's pool at budget_pages +
    // headroom_pages, which is a cycle's eviction over the budget the end-of-cycle
    // spill brings it back under. 0 is unbounded and never spills. Either way the
    // pool fills a slab at a time, so a bound costs only what it is used for;
    // slab_pages fixes that slab and 0 derives it.
    PagedBucket(
        int64_t n_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        at::ScalarType dtype,
        at::Device device,
        int64_t budget_pages = 0,
        int64_t headroom_pages = 0,
        int64_t slab_pages = 0
    );
    ~PagedBucket();
    // Movable, not copyable.
    PagedBucket(PagedBucket&&) noexcept;
    PagedBucket& operator=(PagedBucket&&) noexcept;

    // A paged bucket needs no external resource, so it is always available.
    bool enabled() const {
        return true;
    }

    void begin(int64_t seq);
    void end(int64_t seq);

    // The sequence's pages as a page-index cursor, ascending. token_ids stay in RAM.
    BucketView view(int64_t seq) const;

    // Host-only half of an accept: pop the destination slots and insert the page
    // descriptors, no copy. A bounded pool carries a cycle's eviction headroom over
    // its budget, so this only ever fails on a genuine overrun.
    Reservation reserve(int64_t seq, const DemotedKV& demoted, int64_t step);
    // The copy half: one device-side gather puts the cycle's victim pages in pool
    // layout, then each page's K/V moves STRAIGHT into its (arbitrary) bucket page as a
    // single direct D2H. No host-side staging and no per-page temporary.
    // ASYNCHRONOUS on a device source: the transfers run on a held side stream and this
    // returns with them in flight, so the cycle's host work covers them. wait_fill()
    // fences them and every read of this bucket's KV must follow one.
    void fill(int64_t seq, const ActiveBuffer& active, const Reservation& r);
    // Fence an in-flight fill, on the host and on the current stream: after it the pool
    // holds the pages and the victim slots the gather read are free to be overwritten.
    // A no-op when nothing is in flight.
    void wait_fill();
    // Take a page from the bucket above (a spill), K/V on any device.
    void accept(int64_t seq, const PageKV& page);

    // The given held FULL pages' stored K, pool dtype and pool device; see GatheredK.
    // Rows come out in POOL SLOT order, so a page list crossing slabs still gathers
    // every byte exactly once. The slab plan depends only on page_ids, so it is built
    // and uploaded once for all layers. Pages must be held.
    GatheredK gather_k(int64_t seq, const std::vector<int64_t>& page_ids, const std::vector<int64_t>& layers) const;

    // Promote: remove the given pages and return their full KV
    // ([cnt, n_layers, n_kv_heads, head_dim] each) in page_ids order, as VIEWS into the
    // pool -- no copy and no restage. The slots are marked dead and freed at the next
    // release_pending, so the views stay valid for the rest of the cycle.
    std::vector<PageKV> take_pages(int64_t seq, const std::vector<int64_t>& page_ids);

    // Remove and return the coldest pages until the sequence holds <=
    // budget_tokens. Coldest = lowest lazy-decayed effective mass = mass *
    // (1-decay)^(step - mass_step) (decay is the forgetting rate). budget_tokens <=
    // 0 or decay >= 1.0 disables (returns empty). Already within budget => empty.
    std::vector<PageKV> spill_to_budget(int64_t seq, int64_t budget_tokens, int64_t step, double decay);

    // End-of-cycle: return every slot marked dead this cycle to the free list.
    void release_pending(int64_t seq);

    int64_t bucket_tokens(int64_t seq) const;  // tokens held for a seq
    int64_t size() const;  // total across sequences

    bool has_seq(int64_t seq) const;
    bool has_page(int64_t seq, int64_t page_id) const;
    // Pages the sequence's pool is allocated for (>= its live pages).
    int64_t capacity_pages(int64_t seq) const;

  private:
    BucketPool& pool(int64_t seq);
    const BucketPool& pool(int64_t seq) const;

    int64_t n_layers;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t page_size;
    at::ScalarType dtype;
    at::Device device;
    int64_t max_pages;  // budget + headroom; 0 => unbounded
    int64_t slab_pages;  // growth slab; 0 => derived from a byte budget
    std::unordered_map<int64_t, std::unique_ptr<BucketPool>> seqs;
    // Built on the first fill off a device source, then held: a stream and an event
    // taken per cycle would cost more than the overlap they buy, and would show up as
    // host time rather than anywhere on the kernel timeline. Shared by every sequence,
    // which is sound because cycles are serialized.
    std::unique_ptr<FillStream> fill_side;
};

}  // namespace pulsar
