#include "pulsar/runtime/kv/paged_pool.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <numeric>

// PagedPool: the page-major K/V storage + slot free list the RecallBuffer and
// PagedBucket share. Allocation and page arithmetic only; page identity lives in the
// owning component.

namespace pulsar {

namespace {

// A growth allocates one slab, so the slab size is what one growth costs. It is a
// BYTE budget: page-locking host memory runs at a few GB/s, so this bounds a single
// cudaHostAlloc to a few milliseconds. The page cap keeps a pool of tiny pages
// slabbed rather than collapsing into one block.
constexpr int64_t kSlabBytes = 32 << 20;
constexpr int64_t kSlabPagesCap = 256;

int64_t derive_slab_pages(int64_t page_bytes) {
    return std::clamp<int64_t>(kSlabBytes / std::max<int64_t>(page_bytes, 1), 1, kSlabPagesCap);
}

}  // namespace

namespace {

int64_t resolve_slab_pages(int64_t slab_pages, int64_t page_bytes, int64_t max_pages, bool preallocate) {
    if (preallocate) {
        return max_pages;
    }
    const int64_t want = slab_pages > 0 ? slab_pages : derive_slab_pages(page_bytes);
    return max_pages > 0 ? std::min(want, max_pages) : want;
}

}  // namespace

PagedPool::PagedPool(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    at::ScalarType dtype,
    at::Device device,
    bool store_v,
    int64_t max_pages,
    bool pinned,
    bool preallocate,
    int64_t slab_pages
)
    : n_layers(n_layers),
      n_kv_heads(n_kv_heads),
      head_dim(head_dim),
      page_size(page_size),
      dtype(dtype),
      device(device),
      store_v(store_v),
      pinned(pinned && device.is_cpu()),
      slab_pages(resolve_slab_pages(
          slab_pages,
          page_size * n_layers * n_kv_heads * head_dim * at::elementSize(dtype),
          max_pages,
          preallocate
      )),
      max_pages(max_pages > 0 ? (max_pages + this->slab_pages - 1) / this->slab_pages * this->slab_pages : 0) {
    TORCH_CHECK(!preallocate || max_pages > 0, "PagedPool: a preallocated pool needs a page bound");
    if (preallocate) {
        this->add_slab();
    }
}

at::Tensor PagedPool::page_k(int64_t pool_page) const {
    return this->slabs_k[pool_page / this->slab_pages].select(0, pool_page % this->slab_pages);
}

at::Tensor PagedPool::page_v(int64_t pool_page) const {
    TORCH_CHECK(this->store_v, "PagedPool: this pool stores no V");
    return this->slabs_v[pool_page / this->slab_pages].select(0, pool_page % this->slab_pages);
}

at::Tensor PagedPool::layer_k(int64_t layer) const {
    TORCH_CHECK(
        this->slabs_k.size() == 1,
        "PagedPool::layer_k: the pool holds ",
        this->slabs_k.size(),
        " slabs, not one whole-pool block"
    );
    return this->slabs_k[0].select(2, layer);
}

SlabGather PagedPool::plan_gather(const std::vector<int64_t>& pool_pages) const {
    SlabGather plan;
    plan.n_pages = static_cast<int64_t>(pool_pages.size());
    if (plan.n_pages == 0) {
        return plan;
    }
    plan.order.resize(pool_pages.size());
    std::iota(plan.order.begin(), plan.order.end(), 0);
    // Slot order puts one slab's rows in one run, so each group is a contiguous block
    // of the output and no gathered row is ever moved twice.
    std::sort(plan.order.begin(), plan.order.end(), [&](int64_t a, int64_t b) {
        return pool_pages[a] < pool_pages[b];
    });
    auto rows_opts = at::TensorOptions().dtype(at::kLong);
    std::vector<int64_t> rows;
    int64_t slab = -1;
    int64_t base = 0;
    auto flush = [&] {
        if (rows.empty()) {
            return;
        }
        plan.groups.push_back(SlabGather::Group{slab, base, at::tensor(rows, rows_opts).to(this->device)});
        base += static_cast<int64_t>(rows.size());
        rows.clear();
    };
    for (int64_t i : plan.order) {
        const int64_t pool_page = pool_pages[i];
        const int64_t s = pool_page / this->slab_pages;
        if (s != slab) {
            flush();
            slab = s;
        }
        rows.push_back(pool_page % this->slab_pages);
    }
    flush();
    return plan;
}

at::Tensor PagedPool::gather_layer_k(int64_t layer, const SlabGather& plan) const {
    at::Tensor out = at::empty(
        {plan.n_pages, this->page_size, this->n_kv_heads, this->head_dim},
        at::TensorOptions().dtype(this->dtype).device(this->device)
    );
    for (const SlabGather::Group& g : plan.groups) {
        at::Tensor block = out.narrow(0, g.base, g.rows.size(0));
        at::index_select_out(block, this->slabs_k[g.slab].select(2, layer), 0, g.rows);
    }
    return out;
}

void PagedPool::add_slab() {
    auto opts = at::TensorOptions().dtype(this->dtype).device(this->device).pinned_memory(this->pinned);
    const std::vector<int64_t>
        shape{this->slab_pages, this->page_size, this->n_layers, this->n_kv_heads, this->head_dim};
    this->slabs_k.push_back(at::empty(shape, opts));
    if (this->store_v) {
        this->slabs_v.push_back(at::empty(shape, opts));
    }
    const int64_t base = this->capacity;
    this->capacity += this->slab_pages;
    for (int64_t p = base; p < this->capacity; ++p) {
        this->free_pages.push_back(p);
    }
}

int64_t PagedPool::alloc_page() {
    if (this->free_pages.empty()) {
        TORCH_CHECK(
            this->max_pages <= 0 || this->capacity < this->max_pages,
            "PagedPool: bounded pool of ",
            this->capacity,
            " pages is full; raise its token budget"
        );
        this->add_slab();
    }
    const int64_t p = this->free_pages.back();
    this->free_pages.pop_back();
    return p;
}

}  // namespace pulsar
