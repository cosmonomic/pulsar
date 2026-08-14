#include "pulsar/runtime/kv/paged_bucket.hpp"

#include "host_pages.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <optional>
#include <utility>

// PagedBucket: one PagedPool + descriptor map per sequence. Accepts whole pages
// from the ActiveBuffer (reserve + fill) or from the bucket above (a spill), serves
// its page cursor for history/recall candidates, gathers candidate/promoted pages
// out of the pool, and releases its coldest pages on a token budget. Sees no other
// bucket; the KVMemory routes what comes out.

namespace pulsar {

// One sequence's storage: the flat pool, the page-index descriptor map, and the
// slots this cycle marked dead (returned to the pool by release_pending).
struct BucketPool {
    BucketPool(
        int64_t n_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        at::ScalarType dtype,
        at::Device device,
        int64_t max_pages,
        int64_t slab_pages
    )
        : pool(
              n_layers,
              n_kv_heads,
              head_dim,
              page_size,
              dtype,
              device,
              /*store_v=*/true,
              max_pages,
              /*pinned=*/true,
              /*preallocate=*/false,
              slab_pages
          ) {}

    int64_t token_count() const {
        int64_t n = 0;
        for (const auto& [pid, d] : this->pages) {
            n += d.cnt();
        }
        return n;
    }

    const PoolPage& at(int64_t page_id) const {
        auto it = this->pages.find(page_id);
        TORCH_CHECK(it != this->pages.end(), "PagedBucket: page ", page_id, " not held");
        return it->second;
    }

    // Remove a page's descriptor, returning its KV as pool VIEWS. The slot is only
    // marked dead, so those views stay valid until release_pending.
    PageKV take(int64_t page_id) {
        auto it = this->pages.find(page_id);
        TORCH_CHECK(it != this->pages.end(), "PagedBucket: page ", page_id, " not held");
        PoolPage& d = it->second;
        const int64_t cnt = d.cnt();
        PageKV out;
        out.page_id = d.page_id;
        out.token_ids = at::tensor(d.token_ids, at::TensorOptions().dtype(at::kLong));
        out.mass = d.mass;
        out.mass_step = d.mass_step;
        out.k = this->pool.page_k(d.pool_page).narrow(0, 0, cnt);
        out.v = this->pool.page_v(d.pool_page).narrow(0, 0, cnt);
        this->pending_free.push_back(d.pool_page);
        this->pages.erase(it);
        return out;
    }

    PagedPool pool;
    std::unordered_map<int64_t, PoolPage> pages;  // page_id -> slot + metadata
    std::vector<int64_t> pending_free;  // slots dead this cycle
};

// The side stream a fill's gather and transfers run on, with the event it joins the
// evicting stream on and the one that fences it. Pooled and held; see PagedBucket.
struct FillStream {
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAEvent evicted;  // the eviction's rotations are done
    at::cuda::CUDAEvent done;  // the fill's transfers are done
    bool pending = false;
};

PagedBucket::PagedBucket(
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    at::ScalarType dtype,
    at::Device device,
    int64_t budget_pages,
    int64_t headroom_pages,
    int64_t slab_pages
)
    : n_layers(n_layers),
      n_kv_heads(n_kv_heads),
      head_dim(head_dim),
      page_size(page_size),
      dtype(dtype),
      device(device),
      max_pages(budget_pages > 0 ? budget_pages + headroom_pages : 0),
      slab_pages(slab_pages) {}

PagedBucket::~PagedBucket() = default;
PagedBucket::PagedBucket(PagedBucket&&) noexcept = default;
PagedBucket& PagedBucket::operator=(PagedBucket&&) noexcept = default;

BucketPool& PagedBucket::pool(int64_t seq) {
    auto it = this->seqs.find(seq);
    TORCH_CHECK(it != this->seqs.end(), "PagedBucket: seq ", seq, " not begun");
    return *it->second;
}

const BucketPool& PagedBucket::pool(int64_t seq) const {
    auto it = this->seqs.find(seq);
    TORCH_CHECK(it != this->seqs.end(), "PagedBucket: seq ", seq, " not begun");
    return *it->second;
}

void PagedBucket::begin(int64_t seq) {
    TORCH_CHECK(this->seqs.find(seq) == this->seqs.end(), "PagedBucket: seq ", seq, " already begun");
    this->seqs.emplace(
        seq,
        std::make_unique<BucketPool>(
            this->n_layers,
            this->n_kv_heads,
            this->head_dim,
            this->page_size,
            this->dtype,
            this->device,
            this->max_pages,
            this->slab_pages
        )
    );
}

void PagedBucket::end(int64_t seq) {
    this->seqs.erase(seq);
}

bool PagedBucket::has_seq(int64_t seq) const {
    return this->seqs.find(seq) != this->seqs.end();
}

bool PagedBucket::has_page(int64_t seq, int64_t page_id) const {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return false;
    }
    return it->second->pages.count(page_id) > 0;
}

int64_t PagedBucket::capacity_pages(int64_t seq) const {
    return this->pool(seq).pool.capacity;
}

BucketView PagedBucket::view(int64_t seq) const {
    BucketView v;
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return v;
    }
    const BucketPool& p = *it->second;
    std::vector<int64_t> ids;
    ids.reserve(p.pages.size());
    for (const auto& [pid, d] : p.pages) {
        ids.push_back(pid);
    }
    std::sort(ids.begin(), ids.end());  // address order, materialized on demand
    v.pages.reserve(ids.size());
    for (int64_t pid : ids) {
        v.pages.push_back(BucketPageRef{pid, &p.pages.at(pid).token_ids});
    }
    return v;
}

Reservation PagedBucket::reserve(int64_t seq, const DemotedKV& demoted, int64_t step) {
    Reservation r;
    if (demoted.num_tokens() == 0) {
        return r;
    }
    BucketPool& p = this->pool(seq);
    const int64_t ps = this->page_size;
    std::vector<DemotedPage> pages = demoted_pages(demoted, ps, "PagedBucket::reserve");
    const size_t n_pages = pages.size();
    r.page_ids.resize(n_pages);
    r.src_bases.resize(n_pages);
    r.dst_bases.resize(n_pages);
    for (size_t pi = 0; pi < n_pages; ++pi) {
        DemotedPage& src = pages[pi];
        // A page lives in exactly one place, so it can never be reserved while
        // already held.
        TORCH_CHECK(
            p.pages.find(src.page_id) == p.pages.end(),
            "PagedBucket::reserve: page ",
            src.page_id,
            " already held for seq ",
            seq
        );
        PoolPage d;
        d.page_id = src.page_id;
        d.pool_page = p.pool.alloc_page();
        d.token_ids = std::move(src.token_ids);
        d.mass = src.mass;
        d.mass_step = step;
        r.page_ids[pi] = d.page_id;
        r.src_bases[pi] = src.src_base;
        r.dst_bases[pi] = d.pool_page;
        p.pages.emplace(d.page_id, std::move(d));
    }
    return r;
}

namespace {

// The evicted pages' K (or V) out of the ActiveBuffer's layer-major pool, gathered
// into one contiguous [n_pages, page_size, n_layers, n_kv_heads, head_dim] device
// block: the pool layout, so each page's slice is the exact bytes its destination
// takes. One gather serves the whole cycle; ATen would otherwise make the same
// staging copy per page.
at::Tensor gather_victim_pages(
    const at::Tensor& pool,
    const at::Tensor& slots,
    int64_t n_pages,
    int64_t page_size,
    int64_t n_layers,
    int64_t n_kv_heads,
    int64_t head_dim
) {
    return pool.view({n_layers, -1, n_kv_heads, head_dim})
        .transpose(0, 1)
        .index_select(0, slots)
        .view({n_pages, page_size, n_layers, n_kv_heads, head_dim});
}

}  // namespace

void PagedBucket::fill(int64_t seq, const ActiveBuffer& active, const Reservation& r) {
    if (r.page_ids.empty()) {
        return;
    }
    this->wait_fill();  // one fill in flight at a time
    BucketPool& p = this->pool(seq);
    const int64_t ps = this->page_size;
    const int64_t n = static_cast<int64_t>(r.page_ids.size());
    std::vector<int64_t> slots;
    slots.reserve(n * ps);
    for (int64_t base : r.src_bases) {
        for (int64_t o = 0; o < ps; ++o) {
            slots.push_back(base + o);
        }
    }
    const bool async = active.device().is_cuda();
    if (async && !this->fill_side) {
        this->fill_side = std::make_unique<FillStream>();
    }
    // The gather reads slots the eviction's rotations just wrote on the current stream,
    // so the side stream joins it before reading the pool.
    std::optional<at::cuda::CUDAStreamGuard> guard;
    if (async) {
        this->fill_side->evicted.record(at::cuda::getCurrentCUDAStream());
        this->fill_side->evicted.block(this->fill_side->stream);
        guard.emplace(this->fill_side->stream);
    }
    at::Tensor idx = at::tensor(slots, at::TensorOptions().dtype(at::kLong)).to(active.device());
    at::Tensor
        kstage = gather_victim_pages(active.k_pool_all(), idx, n, ps, this->n_layers, this->n_kv_heads, this->head_dim);
    at::Tensor
        vstage = gather_victim_pages(active.v_pool_all(), idx, n, ps, this->n_layers, this->n_kv_heads, this->head_dim);
    // Both sides contiguous and the host pool page-locked, so each page is one direct
    // D2H with no bounce buffer and no host-side scatter.
    for (int64_t pi = 0; pi < n; ++pi) {
        p.pool.page_k(r.dst_bases[pi]).copy_(kstage.select(0, pi), async);
        p.pool.page_v(r.dst_bases[pi]).copy_(vstage.select(0, pi), async);
    }
    if (async) {
        this->fill_side->done.record(this->fill_side->stream);
        this->fill_side->pending = true;
    }
}

void PagedBucket::wait_fill() {
    if (!this->fill_side || !this->fill_side->pending) {
        return;
    }
    // The pool is host memory, so a host read has to wait as well as the stream that
    // will overwrite the victim slots.
    this->fill_side->done.block(at::cuda::getCurrentCUDAStream());
    this->fill_side->done.synchronize();
    this->fill_side->pending = false;
}

void PagedBucket::accept(int64_t seq, const PageKV& page) {
    BucketPool& p = this->pool(seq);
    const int64_t cnt = page.token_ids.numel();
    TORCH_CHECK(
        p.pages.find(page.page_id) == p.pages.end(),
        "PagedBucket: page ",
        page.page_id,
        " already held for seq ",
        seq
    );
    PoolPage d;
    d.page_id = page.page_id;
    d.pool_page = p.pool.alloc_page();
    at::Tensor token_ids_c = page.token_ids.to(at::kCPU, at::kLong).contiguous();
    d.token_ids.assign(token_ids_c.data_ptr<int64_t>(), token_ids_c.data_ptr<int64_t>() + token_ids_c.numel());
    d.mass = page.mass;
    d.mass_step = page.mass_step;
    // k/v are already in pool layout [cnt, n_layers, n_kv_heads, head_dim]: one copy
    // each, no restage.
    p.pool.page_k(d.pool_page).narrow(0, 0, cnt).copy_(page.k);
    p.pool.page_v(d.pool_page).narrow(0, 0, cnt).copy_(page.v);
    p.pages.emplace(d.page_id, std::move(d));
}

GatheredK
PagedBucket::gather_k(int64_t seq, const std::vector<int64_t>& page_ids, const std::vector<int64_t>& layers) const {
    const BucketPool& p = this->pool(seq);
    std::vector<int64_t> slots;
    slots.reserve(page_ids.size());
    for (int64_t pid : page_ids) {
        slots.push_back(p.at(pid).pool_page);
    }
    // One slab plan, built and uploaded ONCE for every requested layer, then one
    // whole-page index_select per slab per layer, which already lands in the
    // batched-rerank layout.
    const SlabGather plan = p.pool.plan_gather(slots);
    GatheredK out;
    out.order = plan.order;
    out.layers.reserve(layers.size());
    for (int64_t layer : layers) {
        out.layers.push_back(p.pool.gather_layer_k(layer, plan));
    }
    return out;
}

std::vector<PageKV> PagedBucket::take_pages(int64_t seq, const std::vector<int64_t>& page_ids) {
    if (page_ids.empty()) {
        return {};
    }
    BucketPool& p = this->pool(seq);
    std::vector<PageKV> out;
    out.reserve(page_ids.size());
    for (int64_t pid : page_ids) {
        out.push_back(p.take(pid));
    }
    return out;
}

std::vector<PageKV> PagedBucket::spill_to_budget(int64_t seq, int64_t budget_tokens, int64_t step, double decay) {
    std::vector<PageKV> out;
    if (budget_tokens <= 0 || decay >= 1.0 || !this->has_seq(seq)) {
        return out;
    }
    BucketPool& p = this->pool(seq);
    if (p.token_count() <= budget_tokens) {
        return out;
    }

    // Held page ids, ascending: the stable coldest-first sort then breaks eff-mass
    // ties by ascending page id.
    std::vector<int64_t> ids;
    ids.reserve(p.pages.size());
    for (const auto& [pid, d] : p.pages) {
        ids.push_back(pid);
    }
    std::sort(ids.begin(), ids.end());
    std::vector<PageMass> mass;
    mass.reserve(ids.size());
    for (int64_t pid : ids) {
        const PoolPage& d = p.pages.at(pid);
        mass.push_back(PageMass{d.mass, d.mass_step});
    }
    const std::vector<int64_t> order = coldest_first(mass, step, decay);
    int64_t tokens = p.token_count();
    for (size_t oi = 0; oi < order.size() && tokens > budget_tokens; ++oi) {
        const int64_t pid = ids[order[oi]];
        tokens -= p.pages.at(pid).cnt();
        out.push_back(p.take(pid));
    }
    return out;
}

void PagedBucket::release_pending(int64_t seq) {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return;
    }
    BucketPool& p = *it->second;
    for (int64_t slot : p.pending_free) {
        p.pool.free_page(slot);
    }
    p.pending_free.clear();
}

int64_t PagedBucket::bucket_tokens(int64_t seq) const {
    auto it = this->seqs.find(seq);
    if (it == this->seqs.end()) {
        return 0;
    }
    return it->second->token_count();
}

int64_t PagedBucket::size() const {
    int64_t n = 0;
    for (const auto& [seq, p] : this->seqs) {
        n += p->token_count();
    }
    return n;
}

}  // namespace pulsar
