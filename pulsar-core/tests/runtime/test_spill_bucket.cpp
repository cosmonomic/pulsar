#include "pulsar/runtime/kv/file_bucket.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"

#include "pulsar/runtime/kv/active_buffer.hpp"

#include <ATen/ATen.h>

// torch's logging header defines a CHECK macro; drop it so Catch2's CHECK wins.
#undef CHECK
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// Both spill buckets satisfy the same SpillBucket concept, so the contract they
// share -- accept, the RAM page cursor, the batched candidate gather, promotion,
// and the coldest-first budget shed -- is one templated suite over both. What
// differs has its own cases below: PagedBucket's reserve/fill split and slot
// lifetime, FileBucket's per-sequence file. Host instances, so these run without
// CUDA.

namespace {

using pulsar::ActiveBuffer;
using pulsar::FileBucket;
using pulsar::PagedBucket;
using pulsar::PageKV;

constexpr int64_t NL = 2, NKV = 2, NQ = 2, HD = 4, PAGE = 8;

std::string temp_dir() {
    std::filesystem::path d = std::filesystem::temp_directory_path() /
        ("pulsar_bucket_test_" + std::to_string(::getpid()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(&d)));
    std::filesystem::remove_all(d);
    return d.string();
}

// Construction differs per medium; the templated cases only ever touch `bucket`.
template <typename B> struct BucketFixture;

template <> struct BucketFixture<PagedBucket> {
    PagedBucket bucket{NL, NKV, HD, PAGE, at::kFloat, at::kCPU};
};

template <> struct BucketFixture<FileBucket> {
    std::string dir = temp_dir();
    FileBucket bucket{NL, NKV, HD, PAGE, at::kFloat, dir};
    ~BucketFixture() {
        std::filesystem::remove_all(this->dir);
    }
};

// One full page whose token ids are all `fill`, with random K/V in pool layout
// [PAGE, NL, NKV, HD].
PageKV page(int64_t page_id, int64_t fill, double mass = 0.0, int64_t mass_step = 0) {
    PageKV pg;
    pg.page_id = page_id;
    pg.token_ids = at::tensor(std::vector<int64_t>(PAGE, fill), at::TensorOptions().dtype(at::kLong));
    pg.mass = mass;
    pg.mass_step = mass_step;
    pg.k = at::randn({PAGE, NL, NKV, HD});
    pg.v = at::randn({PAGE, NL, NKV, HD});
    return pg;
}

std::vector<int64_t> cursor_page_ids(const pulsar::BucketView& view) {
    std::vector<int64_t> out;
    for (const pulsar::BucketPageRef& p : view.pages) {
        out.push_back(p.page_id);
    }
    return out;
}

}  // namespace

TEMPLATE_TEST_CASE("a spill bucket round-trips a page's KV exactly", "[bucket]", PagedBucket, FileBucket) {
    BucketFixture<TestType> f;
    auto& bucket = f.bucket;
    bucket.begin(0);
    PageKV pg = page(3, /*fill=*/77, /*mass=*/1.0);
    const at::Tensor k = pg.k, v = pg.v;
    bucket.accept(0, pg);

    CHECK(bucket.has_page(0, 3));
    CHECK(bucket.bucket_tokens(0) == PAGE);
    CHECK(bucket.size() == PAGE);
    // Identity stays in RAM in both media, so the cursor never reads the KV back.
    auto pv = bucket.view(0).pages;
    REQUIRE(pv.size() == 1);
    CHECK(pv[0].page_id == 3);
    CHECK(*pv[0].token_ids == std::vector<int64_t>(PAGE, 77));

    // gather_k lays out only the requested layers, in `layers` order.
    pulsar::GatheredK g = bucket.gather_k(0, {3}, /*layers=*/{1, 0});
    REQUIRE(g.layers.size() == 2);
    REQUIRE(g.order == std::vector<int64_t>{0});
    REQUIRE(g.layers[0].sizes() == at::IntArrayRef({1, PAGE, NKV, HD}));
    CHECK(at::equal(g.layers[0].select(0, 0).cpu(), k.select(1, 1)));
    CHECK(at::equal(g.layers[1].select(0, 0).cpu(), k.select(1, 0)));

    std::vector<PageKV> taken = bucket.take_pages(0, {3});
    REQUIRE(taken.size() == 1);
    CHECK(taken[0].page_id == 3);
    CHECK(taken[0].mass == 1.0);  // the eviction-time rank survives the medium
    CHECK(at::equal(taken[0].k.cpu(), k));
    CHECK(at::equal(taken[0].v.cpu(), v));
    CHECK_FALSE(bucket.has_page(0, 3));
    CHECK(bucket.bucket_tokens(0) == 0);
    bucket.end(0);
}

TEMPLATE_TEST_CASE(
    "a spill bucket's cursor is page-index sorted and its gather "
    "reports the request position of every row",
    "[bucket]",
    PagedBucket,
    FileBucket
) {
    BucketFixture<TestType> f;
    auto& bucket = f.bucket;
    bucket.begin(7);
    // Accepted out of order: the cursor must present address (== page index) order,
    // because the history merge walks it by index.
    std::vector<at::Tensor> layer0;
    for (int64_t page_id : {5, 1, 3}) {
        PageKV pg = page(page_id, /*fill=*/page_id);
        layer0.push_back(pg.k.select(1, 0));
        bucket.accept(7, pg);
    }
    CHECK(cursor_page_ids(bucket.view(7)) == std::vector<int64_t>{1, 3, 5});

    // The gather is in the bucket's own storage order, which is neither page_ids
    // order nor cursor order: recall matches the rows back through `order`, so a row
    // paired with the wrong request position would silently mis-score every page.
    pulsar::GatheredK g = bucket.gather_k(7, {5, 1}, /*layers=*/{0});
    REQUIRE(g.layers[0].sizes() == at::IntArrayRef({2, PAGE, NKV, HD}));
    REQUIRE(g.order.size() == 2);
    const std::vector<at::Tensor> want{layer0[0], layer0[1]};  // page 5, then page 1
    for (int64_t row = 0; row < 2; ++row) {
        CHECK(at::equal(g.layers[0].select(0, row).cpu(), want[g.order[row]]));
    }

    std::vector<PageKV> taken = bucket.take_pages(7, {3});
    REQUIRE(taken.size() == 1);
    CHECK(taken[0].page_id == 3);
    CHECK_FALSE(bucket.has_page(7, 3));
    CHECK(bucket.bucket_tokens(7) == 2 * PAGE);
    CHECK(cursor_page_ids(bucket.view(7)) == std::vector<int64_t>{1, 5});
    bucket.end(7);
}

TEMPLATE_TEST_CASE("a spill bucket sheds its coldest pages to a token budget", "[bucket]", PagedBucket, FileBucket) {
    BucketFixture<TestType> f;
    auto& bucket = f.bucket;
    bucket.begin(0);
    // Masses descend with the page id, so the shed order is the REVERSE of both the
    // page-index order the cursor presents and the insertion order: a bucket that
    // ignored mass would shed page 0 first and fail here.
    bucket.accept(0, page(0, 10, /*mass=*/0.9));
    bucket.accept(0, page(1, 11, /*mass=*/0.5));
    bucket.accept(0, page(2, 12, /*mass=*/0.2));

    std::vector<PageKV> out = bucket.spill_to_budget(
        0,
        /*budget_tokens=*/PAGE,
        /*step=*/0,
        /*decay=*/0.0
    );
    REQUIRE(out.size() == 2);  // two pages over budget, shed coldest first
    CHECK(out[0].page_id == 2);
    CHECK(out[1].page_id == 1);
    CHECK(out[0].mass == 0.2);  // the rank carries across the spill
    CHECK(bucket.bucket_tokens(0) == PAGE);
    CHECK(bucket.has_page(0, 0));  // the hottest page survived
    // Already within budget, and the two disabling arguments, all shed nothing.
    CHECK(bucket.spill_to_budget(0, PAGE, 0, 0.0).empty());
    CHECK(bucket.spill_to_budget(0, 0, 0, 0.0).empty());
    CHECK(bucket.spill_to_budget(0, 0, 0, /*decay=*/1.0).empty());
    bucket.end(0);
}

TEMPLATE_TEST_CASE("a spill bucket ranks by lazy-decayed mass, not raw mass", "[bucket]", PagedBucket, FileBucket) {
    // The shed ranks by mass * (1 - decay)^(step - mass_step), so a page that was hot
    // long ago is colder than a lukewarm page from this step. With raw mass alone the
    // stale page would survive and the fresh one would spill.
    BucketFixture<TestType> f;
    auto& bucket = f.bucket;
    bucket.begin(0);
    bucket.accept(0, page(0, 10, /*mass=*/0.9, /*mass_step=*/0));  // 0.9 * 0.5^10
    bucket.accept(0, page(1, 11, /*mass=*/0.5, /*mass_step=*/10));  // 0.5

    std::vector<PageKV> out = bucket.spill_to_budget(
        0,
        /*budget_tokens=*/PAGE,
        /*step=*/10,
        /*decay=*/0.5
    );
    REQUIRE(out.size() == 1);
    CHECK(out[0].page_id == 0);  // the stale high-mass page, not the fresh one
    CHECK(bucket.has_page(0, 1));
    bucket.end(0);
}

TEST_CASE("PagedBucket reserve is host-only and fill copies the victim slots", "[bucket]") {
    // Two whole address-aligned victim pages taken straight out of an ActiveBuffer
    // pool. ActiveBuffer::evict itself needs CUDA (the re-rope kernel), so the
    // DemotedKV is assembled here to exercise the reserve/fill split on the host.
    ActiveBuffer active(
        NL,
        /*num_pages=*/8,
        PAGE,
        NKV,
        NQ,
        HD,
        "float32",
        "cpu",
        /*rope_theta=*/1e6
    );
    active.allocate(0, 2 * PAGE);
    std::vector<int64_t> addr(2 * PAGE), toks(2 * PAGE), slots(2 * PAGE);
    for (int64_t i = 0; i < 2 * PAGE; ++i) {
        addr[i] = i;
        toks[i] = 500 + i;
    }
    active.set_identity(0, 0, addr, toks);
    at::Tensor slot_map = active.slot_mapping(0, 0, 2 * PAGE).to(at::kLong);
    for (int64_t i = 0; i < 2 * PAGE; ++i) {
        slots[i] = slot_map[i].item<int64_t>();
    }
    // Distinguishable K/V at every victim slot, so fill's copy is checkable.
    for (int64_t l = 0; l < NL; ++l) {
        at::Tensor kf = active.k_pool(l).view({-1, NKV, HD});
        at::Tensor vf = active.v_pool(l).view({-1, NKV, HD});
        for (int64_t i = 0; i < 2 * PAGE; ++i) {
            kf.select(0, slots[i]).fill_(static_cast<double>(l * 100 + i));
            vf.select(0, slots[i]).fill_(static_cast<double>(-(l * 100 + i)));
        }
    }
    auto i64v = [](const std::vector<int64_t>& v) { return at::tensor(v, at::TensorOptions().dtype(at::kLong)); };
    pulsar::DemotedKV demoted;
    demoted.addr_data = i64v(addr);
    demoted.tok_data = i64v(toks);
    demoted.mass_data = at::tensor(std::vector<float>{0.25f, 0.5f}, at::TensorOptions().dtype(at::kFloat));
    demoted.victim_slots = i64v(slots);

    PagedBucket b(NL, NKV, HD, PAGE, at::kFloat, at::kCPU);
    b.begin(0);
    pulsar::Reservation r = b.reserve(0, demoted, /*step=*/5);
    // reserve is host-only: descriptors and token ids exist before any KV moves.
    REQUIRE(r.page_ids == std::vector<int64_t>{0, 1});
    CHECK(b.has_page(0, 0));
    CHECK(b.has_page(0, 1));
    CHECK(b.bucket_tokens(0) == 2 * PAGE);
    auto pv = b.view(0).pages;
    REQUIRE(pv.size() == 2);
    CHECK(*pv[0].token_ids == std::vector<int64_t>(toks.begin(), toks.begin() + PAGE));
    CHECK(*pv[1].token_ids == std::vector<int64_t>(toks.begin() + PAGE, toks.end()));

    b.fill(0, active, r);
    // Page 1 (addresses [PAGE, 2*PAGE)), layer 1: row i holds l*100 + (PAGE + i).
    pulsar::GatheredK g = b.gather_k(0, {1}, /*layers=*/{1});
    REQUIRE(g.layers[0].sizes() == at::IntArrayRef({1, PAGE, NKV, HD}));
    for (int64_t i = 0; i < PAGE; ++i) {
        CHECK(g.layers[0][0][i][0][0].item<float>() == static_cast<float>(100 + PAGE + i));
    }
    std::vector<PageKV> taken = b.take_pages(0, {0});
    REQUIRE(taken.size() == 1);
    CHECK(taken[0].mass == 0.25);  // the eviction-time rank carries into the bucket
    CHECK(taken[0].v[3][0][0][0].item<float>() == -3.0f);
    b.end(0);
}

TEST_CASE("PagedBucket defers slot frees to release_pending", "[bucket]") {
    // A pool bounded at exactly 2 pages: taking one page must NOT make its slot
    // reusable until release_pending runs, so a third accept before the release
    // hits the budget wall rather than silently reusing a slot a reader may hold.
    PagedBucket b(NL, NKV, HD, PAGE, at::kFloat, at::kCPU, /*budget_pages=*/2);
    b.begin(0);
    CHECK(b.capacity_pages(0) == 0);  // a bound is a cap, not a preallocation
    b.accept(0, page(0, 10));
    b.accept(0, page(1, 11));
    CHECK(b.capacity_pages(0) == 2);
    b.take_pages(0, {0});
    CHECK(b.bucket_tokens(0) == PAGE);  // one page left
    CHECK_THROWS(b.accept(0, page(2, 12)));  // slot still held for the cycle
    b.release_pending(0);
    b.accept(0, page(2, 12));  // now the freed slot is reusable
    CHECK(b.has_page(0, 2));
    CHECK(b.capacity_pages(0) == 2);  // never grew
    b.end(0);
}

TEST_CASE("PagedBucket grows by appending slabs and gathers across them", "[bucket]") {
    // Two pages per slab, so five pages span three slabs. Growth appends: it copies
    // nothing, so KV handed out before it stays valid, and a gather spanning slabs
    // must still pair every row with the page it was asked for.
    constexpr int64_t SLAB = 2;
    PagedBucket b(
        NL,
        NKV,
        HD,
        PAGE,
        at::kFloat,
        at::kCPU,
        /*budget_pages=*/0,
        /*headroom_pages=*/0,
        /*slab_pages=*/SLAB
    );
    b.begin(0);
    std::vector<at::Tensor> layer0;
    for (int64_t page_id : {0, 1}) {
        PageKV pg = page(page_id, /*fill=*/page_id);
        layer0.push_back(pg.k.select(1, 0));
        b.accept(0, pg);
    }
    CHECK(b.capacity_pages(0) == SLAB);
    // Taken but not released, so this is a live view into the first slab.
    std::vector<PageKV> taken = b.take_pages(0, {0});
    for (int64_t page_id : {2, 3, 4}) {
        PageKV pg = page(page_id, /*fill=*/page_id);
        layer0.push_back(pg.k.select(1, 0));
        b.accept(0, pg);
    }
    CHECK(b.capacity_pages(0) == 3 * SLAB);
    CHECK(at::equal(taken[0].k.select(1, 0), layer0[0]));

    // Requested in an order matching neither page id nor slot.
    const std::vector<int64_t> want{4, 1, 3, 2};
    pulsar::GatheredK g = b.gather_k(0, want, /*layers=*/{0});
    REQUIRE(g.layers[0].sizes() == at::IntArrayRef({4, PAGE, NKV, HD}));
    std::vector<int64_t> sorted_order = g.order;
    std::sort(sorted_order.begin(), sorted_order.end());
    CHECK(sorted_order == std::vector<int64_t>{0, 1, 2, 3});
    for (int64_t row = 0; row < 4; ++row) {
        CHECK(at::equal(g.layers[0].select(0, row), layer0[want[g.order[row]]]));
    }
    b.end(0);
}

TEST_CASE("FileBucket deletes its per-sequence file on end", "[bucket]") {
    std::string dir = temp_dir();
    {
        FileBucket store(NL, NKV, HD, PAGE, at::kFloat, dir);
        store.begin(0);
        store.accept(0, page(3, /*fill=*/100));
        CHECK(std::filesystem::exists(std::filesystem::path(dir) / "0"));
        store.end(0);
    }
    CHECK_FALSE(std::filesystem::exists(std::filesystem::path(dir) / "0"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("FileBucket is disabled when its directory is empty", "[bucket]") {
    FileBucket store(NL, NKV, HD, PAGE, at::kFloat, "");
    CHECK_FALSE(store.enabled());
    store.begin(0);  // no-op: no file is ever created
    CHECK_FALSE(store.has_seq(0));
    CHECK(store.size() == 0);
}
