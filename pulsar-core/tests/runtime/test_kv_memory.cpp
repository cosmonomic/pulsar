#include "pulsar/runtime/kv/kv_memory.hpp"
#include "pulsar/runtime/kv/active_buffer.hpp"
#include "pulsar/runtime/kv/file_bucket.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"
#include "pulsar/runtime/kv/recall.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <unordered_set>
#include <vector>

// The PageIndexed/SpillBucket concepts and the address cursors over the ActiveBuffer
// and the bucket chain; the active/history PageViews across all of them; the s1 -> s2
// spill routing; and the s1-union-s2 recall that promotes a selected page out of
// whichever bucket holds it. Host-only.

namespace {

using pulsar::ActiveBuffer;
using pulsar::ActiveView;
using pulsar::AttnScoreRetriever;
using pulsar::BucketView;
using pulsar::FileBucket;
using pulsar::KVMemory;
using pulsar::PagedBucket;
using pulsar::PageKV;
using pulsar::Payload;
using pulsar::RecallBuffer;
using pulsar::RecallConfig;
using pulsar::RecallCounts;
using pulsar::RecallParams;

constexpr int64_t PS = 16;
constexpr double THETA = 1'000'000.0;
constexpr int64_t TOK0 = 1000;  // token id of address a is TOK0 + a
constexpr int64_t STAGE_PAGES = 16;  // staging capacity these tests never exceed

// Flatten a PageView (dogfoods std::views::values) to its token-id stream.
std::vector<int64_t> flatten(const pulsar::PageView& pv) {
    std::vector<int64_t> out;
    for (const pulsar::PageRef& page : pv | std::views::values) {
        for (int64_t t : page.token_ids()) {
            out.push_back(t);
        }
    }
    return out;
}
// The PageView's page indices (dogfoods std::views::keys).
std::vector<int64_t> page_indices(const pulsar::PageView& pv) {
    std::vector<int64_t> out;
    for (int64_t k : pv | std::views::keys) {
        out.push_back(k);
    }
    return out;
}

ActiveBuffer cpu_alloc() {
    return ActiveBuffer(
        /*n_layers=*/1,
        /*num_pages=*/32,
        PS,
        /*n_kv_heads=*/1,
        /*n_q_heads=*/1,
        /*head_dim=*/4,
        "float32",
        "cpu",
        THETA
    );
}
RecallBuffer cpu_buffer() {
    return RecallBuffer(
        /*n_rel_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PS,
        std::string("float32"),
        STAGE_PAGES,
        /*max_sessions=*/1
    );
}
PagedBucket cpu_s1() {
    return PagedBucket(
        /*n_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PS,
        at::kFloat,
        at::kCPU
    );
}
std::string temp_dir() {
    std::filesystem::path d = std::filesystem::temp_directory_path() /
        ("pulsar_kvmem_" + std::to_string(::getpid()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(&d)));
    std::filesystem::remove_all(d);
    return d.string();
}
FileBucket cpu_s2(const std::string& dir) {
    return FileBucket(
        /*n_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PS,
        at::kFloat,
        dir
    );
}

// One aligned page at [addr0, addr0+PS): token id = TOK0+addr, the given per-token
// K/V [PS, n_kv_heads, head_dim] of the single layer, and a per-page mass.
PageKV make_page(int64_t addr0, const at::Tensor& k, const at::Tensor& v, double mass) {
    std::vector<int64_t> tok(PS);
    for (int64_t i = 0; i < PS; ++i) {
        tok[i] = TOK0 + addr0 + i;
    }
    PageKV pg;
    pg.page_id = addr0 / PS;
    pg.token_ids = at::tensor(tok, at::TensorOptions().dtype(at::kLong));
    pg.k = k.unsqueeze(1);
    pg.v = v.unsqueeze(1);
    pg.mass = mass;
    return pg;
}

// Record the segment documents for addresses [0, n), the token id of address a
// being TOK0 + a -- what the engine's note_tokens hook lands as tokens are born.
// Documents are residency-independent, so this is independent of which component
// ends up holding each page. segment_size is the grouping, and the recall that reads
// these documents must name the same one.
void note_range(KVMemory& memory, int64_t n, int64_t segment_size = PS) {
    std::vector<int64_t> addr(n), tok(n);
    for (int64_t i = 0; i < n; ++i) {
        addr[i] = i;
        tok[i] = TOK0 + i;
    }
    memory.note_tokens(0, addr, tok, segment_size);
}

// Demote one aligned page into s1, mirroring what ActiveBuffer::evict ->
// accept_evicted lands.
void evict_page(PagedBucket& s1, int64_t addr0, const at::Tensor& k, const at::Tensor& v, float mass) {
    s1.accept(0, make_page(addr0, k, v, mass));
}

// Build a split state for one 64-token conversation: address-pages 0,1 (addresses
// 0..31) demoted to s1, address-pages 2,3 (addresses 32..63) active.
// Random demoted K/V (history reconstruction uses only token ids). Page 1 is the
// COLDER of the two, so a budgeted spill picks it and not the lower page id.
void build_split(ActiveBuffer& a, PagedBucket& s1) {
    s1.begin(0);
    evict_page(s1, 0, at::randn({PS, 1, 4}), at::randn({PS, 1, 4}), 1.0f);
    evict_page(s1, 16, at::randn({PS, 1, 4}), at::randn({PS, 1, 4}), 0.1f);

    a.allocate(0, 32);
    std::vector<int64_t> hi_addr(32), hi_tok(32);
    for (int64_t i = 0; i < 32; ++i) {
        hi_addr[i] = 32 + i;
        hi_tok[i] = TOK0 + 32 + i;
    }
    a.set_identity(0, /*start_pos=*/0, hi_addr, hi_tok);
}

// A [PS, 1, 4] page-K tensor in pool layout (n_kv_heads=1) with every token row set
// to `row`.
at::Tensor page_k(const std::vector<float>& row) {
    at::Tensor t = at::empty({PS, 1, 4}, at::TensorOptions().dtype(at::kFloat));
    for (int64_t i = 0; i < PS; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            t[i][0][j] = row[j];
        }
    }
    return t;
}
// A single-layer query Q [1, 1, 4] (one token, one head).
std::vector<at::Tensor> query_vec(const std::vector<float>& row) {
    at::Tensor t = at::empty({1, 1, 4}, at::TensorOptions().dtype(at::kFloat));
    for (int64_t j = 0; j < 4; ++j) {
        t[0][0][j] = row[j];
    }
    return {t};
}
// The reranker scores a candidate page as sigmoid(page_best_logit - lse +
// log attended_len). These tests hand it lse == 0 and log attended_len == 0, so a page
// scores sigmoid(page_best_logit) -- the reference every scoring_page() below is stated
// against, and independent of the active buffer's contents.
constexpr int64_t N_WORKING = 8;

// Append N_WORKING attended tokens at [addr0, addr0 + N_WORKING) with all-zero K.
// They are the sequence's newest attended tokens, which is the slice recall takes as
// its reference. addr0 must be page-aligned. Returns addr0 + N_WORKING, the first
// address past the window, so a caller can note the documents up to it.
int64_t attend_working(ActiveBuffer& a, int64_t addr0) {
    const int64_t start_pos = a.has_seq(0) ? a.active_len(0) : 0;
    if (a.has_seq(0)) {
        a.append(0, N_WORKING);
    } else {
        a.allocate(0, N_WORKING);
    }
    std::vector<int64_t> addr(N_WORKING), tok(N_WORKING);
    for (int64_t i = 0; i < N_WORKING; ++i) {
        addr[i] = addr0 + i;
        tok[i] = TOK0 + addr0 + i;
    }
    a.set_identity(0, start_pos, addr, tok);
    at::Tensor slots = a.slot_mapping(0, start_pos, N_WORKING).to(at::kLong);
    a.k_pool(0)
        .reshape({-1, 1, 4})
        .index_copy_(0, slots, at::zeros({N_WORKING, 1, 4}, at::TensorOptions().dtype(at::kFloat)));
    return addr0 + N_WORKING;
}

// A neutral reference for one query token on one layer: score == sigmoid(logit).
std::vector<at::Tensor> flat_lse(int64_t q_len = 1) {
    return {at::zeros({q_len, 1}, at::TensorOptions().dtype(at::kFloat))};
}
at::Tensor flat_log_len(int64_t q_len = 1) {
    return at::zeros({q_len}, at::TensorOptions().dtype(at::kFloat));
}
// Query rows at position 0: the rerank's per-row rotation is then the identity and a
// page's score is the plain q.k these fixtures build.
at::Tensor flat_positions(int64_t q_len = 1) {
    return at::zeros({q_len}, at::TensorOptions().dtype(at::kLong));
}

// A page whose rerank score against QUERY_AXIS is `score`, for any score in (0, 1):
// every key sits on the query axis at the logit sigmoid inverts to, so the page's
// max-over-keys logit IS that logit and the score is sigmoid of it.
// L = q.k / sqrt(head_dim), head_dim 4, |q| = 1.
const std::vector<float> QUERY_AXIS{1, 0, 0, 0};
at::Tensor scoring_page(double score) {
    const double logit = std::log(score) - std::log1p(-score);
    at::Tensor t = at::zeros({PS, 1, 4}, at::TensorOptions().dtype(at::kFloat));
    for (int64_t i = 0; i < PS; ++i) {
        t[i][0][0] = static_cast<float>(2.0 * logit);  // scale = 1 / sqrt(4)
    }
    return t;
}
// A recall config over layer 0. Scores are plain multiples of uniform.
RecallConfig recall_cfg() {
    RecallConfig c;
    c.layers = {0};
    c.seed = 0;
    return c;
}

// The per-call session knobs these tests share: bar 0 (keep every scored page), no
// prefilter, no selection sampling and no neighbourhood expansion. segment_size must be
// the grouping note_range recorded the documents under. short_offset only labels the
// promoted payloads here, since these fixtures write bucket K directly.
RecallParams recall_params(int64_t segment_size = PS) {
    RecallParams p;
    p.prefilter_k = -1;
    p.short_offset = PS;
    p.candidate_position = PS;  // scored where stored: the rerank rotates nothing
    p.segment_size = segment_size;
    return p;
}

}  // namespace

TEST_CASE("PageIndexed and SpillBucket concepts hold for the components", "[kvmemory]") {
    STATIC_REQUIRE(pulsar::PageIndexed<ActiveView>);
    STATIC_REQUIRE(pulsar::PageIndexed<BucketView>);
    STATIC_REQUIRE(pulsar::PageCursor<pulsar::ActiveCursor>);
    STATIC_REQUIRE(pulsar::PageCursor<BucketView::Cursor>);
    // Medium-agnostic: either bucket implementation may sit in either position.
    STATIC_REQUIRE(pulsar::SpillBucket<PagedBucket>);
    STATIC_REQUIRE(pulsar::SpillBucket<FileBucket>);
}

TEST_CASE("history reconstructs a conversation across the active buffer and s1", "[kvmemory]") {
    auto a = cpu_alloc();
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    build_split(a, s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());

    std::vector<int64_t> full = flatten(memory.history(0));
    REQUIRE(full.size() == 64);
    for (int64_t a_ = 0; a_ < 64; ++a_) {
        CHECK(full[a_] == TOK0 + a_);
    }
    // A caller slices its own sub-range from the address-ordered stream.
    std::vector<int64_t> mid(full.begin() + 20, full.begin() + 40);
    REQUIRE(mid.size() == 20);
    for (int64_t i = 0; i < 20; ++i) {
        CHECK(mid[i] == TOK0 + 20 + i);
    }
    // active() is the attended set (pages 2,3 -> addr 32..63).
    std::vector<int64_t> ctx = flatten(memory.active(0));
    REQUIRE(ctx.size() == 32);
    for (int64_t i = 0; i < 32; ++i) {
        CHECK(ctx[i] == TOK0 + 32 + i);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("end_cycle routes s1 overflow into s2; history still covers all three", "[kvmemory]") {
    auto a = cpu_alloc();
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    build_split(a, s1);  // demoted: pages 0,1; active: pages 2,3
    s2.begin(0);
    KVMemory memory(a, buf, s1, s2, recall_cfg());

    // Budget = one page: the colder page (index 1, the HIGHER page id) spills to s2,
    // so the choice is the mass ranking and not the page order. The history is then
    // split across all three.
    memory.end_cycle(0, /*ram_bucket_size=*/PS, /*step=*/0, /*decay=*/0.0);
    CHECK(s1.bucket_tokens(0) == PS);  // one page left in s1
    CHECK(s2.bucket_tokens(0) == PS);  // one page in s2
    CHECK(s2.has_page(0, 1));
    CHECK(s1.has_page(0, 0));

    // token_ids stay in RAM in every bucket -> history reconstructs the full stream.
    std::vector<int64_t> full = flatten(memory.history(0));
    REQUIRE(full.size() == 64);
    for (int64_t a_ = 0; a_ < 64; ++a_) {
        CHECK(full[a_] == TOK0 + a_);
    }

    s2.end(0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("the per-component views partition the history; history is their merge", "[kvmemory]") {
    auto a = cpu_alloc();
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    build_split(a, s1);  // demoted: pages 0,1 (addr 0..31); active: pages 2,3
    s2.begin(0);
    KVMemory memory(a, buf, s1, s2, recall_cfg());

    // Spill page 1 (addr 16..31, the colder one) to s2, leaving s1 with page 0. Now
    // the 64-address history is partitioned one page per component.
    memory.end_cycle(0, /*ram_bucket_size=*/PS, /*step=*/0, /*decay=*/0.0);
    REQUIRE(s2.has_page(0, 1));

    auto range = [](int64_t lo, int64_t hi) {
        std::vector<int64_t> v;
        for (int64_t x = lo; x < hi; ++x) {
            v.push_back(TOK0 + x);
        }
        return v;
    };
    // Each component holds its own gappy, address-ordered subset, read via its page
    // cursor -- the same PageIndexed surface for both media.
    std::vector<int64_t> s2_ids;
    for (const auto& p : s2.view(0).pages) {
        s2_ids.insert(s2_ids.end(), p.token_ids->begin(), p.token_ids->end());
    }
    CHECK(s2_ids == range(16, 32));  // s2: page 1
    std::vector<int64_t> s1_ids;
    for (const auto& p : s1.view(0).pages) {
        s1_ids.insert(s1_ids.end(), p.token_ids->begin(), p.token_ids->end());
    }
    CHECK(s1_ids == range(0, 16));  // s1: page 0
    // The active buffer alone is the active view -- exactly the attended set in
    // address (== attention) order.
    CHECK(flatten(memory.active(0)) == range(32, 64));  // active: pages 2,3
    CHECK(static_cast<int64_t>(flatten(memory.active(0)).size()) == a.active_len(0));
    // history is the ordered merge -> the full contiguous stream, its page indices
    // the merged [0,4) in order (dogfoods std::views::keys/values).
    CHECK(flatten(memory.history(0)) == range(0, 64));
    CHECK(page_indices(memory.history(0)) == (std::vector<int64_t>{0, 1, 2, 3}));

    s2.end(0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("exclusive completeness of active and demoted page indices covers history", "[kvmemory]") {
    auto a = cpu_alloc();
    auto s1 = cpu_s1();
    build_split(a, s1);
    std::unordered_set<int64_t> keys;
    const int64_t n0 = a.num_active_pages(0);
    for (int64_t pp = 0; pp < n0; ++pp) {
        CHECK(keys.insert(a.page_id(0, pp)).second);
    }
    for (const auto& p : s1.view(0).pages) {
        CHECK(keys.insert(p.page_id).second);
    }
    const int64_t next = a.next_page_index(0);
    REQUIRE(static_cast<int64_t>(keys.size()) == next);
    for (int64_t q = 0; q < next; ++q) {
        CHECK(keys.count(q) == 1);
    }
}

TEST_CASE("recall promotes an s2 page matching a no-spill s1 run", "[kvmemory]") {
    auto a = cpu_alloc();
    std::string dir = temp_dir();
    // The attended window sits at page 2, just past the two demoted pages.
    const int64_t noted = attend_working(a, /*addr0=*/2 * PS);

    // Page 0 (addr 0..15) reranks at 0.9, page 1 (addr 16..31) at 0.2, so page 0 ranks
    // first and a one-page capacity promotes it.
    at::Tensor k0 = scoring_page(0.9), v0 = page_k({1, 0, 0, 0});
    at::Tensor k1 = scoring_page(0.2), v1 = page_k({0, 1, 0, 0});
    auto page0 = [&](PagedBucket& s) { evict_page(s, 0, k0, v0, /*mass=*/0.0f); };  // cold -> spills first
    auto page1 = [&](PagedBucket& s) { evict_page(s, 16, k1, v1, /*mass=*/1.0f); };
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);

    // Reference: no spill -> recall promotes page 0 out of s1.
    Payload ref;
    {
        auto buf = cpu_buffer();
        auto s1 = cpu_s1();
        auto s2 = cpu_s2(dir + "_ref");
        s1.begin(0);
        page0(s1);
        page1(s1);
        KVMemory memory(a, buf, s1, s2, recall_cfg());
        note_range(memory, noted);
        auto pls = memory.recall(
            0,
            query,
            /*capacity_tokens=*/PS,
            qv,
            flat_lse(),
            flat_log_len(),
            flat_positions(),
            recall_params()
        );
        REQUIRE(pls.size() == 1);
        ref = std::move(pls[0]);
    }

    // Spill run: page 0 lives in s2, recall must read it back from disk.
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2(dir);
    s1.begin(0);
    s2.begin(0);
    page0(s1);
    page1(s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted);
    memory.end_cycle(0, /*ram_bucket_size=*/PS, /*step=*/0, /*decay=*/0.0);
    REQUIRE(s2.has_page(0, 0));  // the aligned page is in s2

    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        recall_params()
    );
    REQUIRE(pls.size() == 1);
    const Payload& got = pls[0];
    // Same identity + byte-exact KV as the no-spill s1 promotion: the medium the page
    // was read from is the only thing that changed.
    CHECK(got.base_addr == ref.base_addr);
    CHECK(at::equal(got.token_ids, ref.token_ids));
    REQUIRE(got.k.size(1) == 1);
    CHECK(at::equal(got.k, ref.k));
    CHECK(at::equal(got.v, ref.v));
    CHECK(at::equal(got.k.select(1, 0),
                    k0));  // the fixed page-0 K, round-tripped through disk
    CHECK_FALSE(s2.has_page(0, 0));  // promotion consumed the s2 page

    s2.end(0);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(dir + "_ref");
}

TEST_CASE("one recall cycle draws candidates and promotions from both buckets", "[kvmemory][recall]") {
    // Four contiguous pages split across the chain: pages 0 and 2 in s2 (disk),
    // pages 1 and 3 in s1 (host RAM). The batched staging must interleave the two
    // sources by scored-page row, and the promotion must take from whichever bucket
    // holds each winner.
    //
    // Rerank scores against the query axis: page 0 -> 0.95, pages 1 and 2 -> 0.3,
    // page 3 -> 0.8. A two-page capacity therefore promotes page 0 (from s2) then page
    // 3 (from s1) -- one winner from each bucket.
    auto a = cpu_alloc();
    // The attended window sits at page 4, just past the four demoted pages.
    const int64_t noted = attend_working(a, /*addr0=*/4 * PS);
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    s1.begin(0);
    s2.begin(0);
    at::Tensor hot = scoring_page(0.95), warm = scoring_page(0.8);
    s2.accept(0, make_page(0, hot, page_k({1, 0, 0, 0}), 1.0));
    s1.accept(0, make_page(16, scoring_page(0.3), page_k({0, 1, 0, 0}), 1.0));
    s2.accept(0, make_page(32, scoring_page(0.3), page_k({0, 1, 0, 0}), 1.0));
    s1.accept(0, make_page(48, warm, page_k({0.6, 0.8, 0, 0}), 1.0));

    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    // segment_size = 4 * PS puts all four pages in one segment, so the prefilter
    // keeps every page and the rerank sees a mixed-source candidate set.
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, /*segment_size=*/4 * PS);
    // The full history merges the four demoted pages and the attended window's page 4.
    CHECK(page_indices(memory.history(0)) == (std::vector<int64_t>{0, 1, 2, 3, 4}));

    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/2 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        recall_params(/*segment_size=*/4 * PS)
    );
    REQUIRE(pls.size() == 2);
    CHECK(pls[0].base_addr == 0);  // best: the aligned page, from s2
    CHECK(pls[1].base_addr == 48);  // second: the warm page, from s1
    // Each winner left the bucket that held it; the losers stayed put.
    CHECK_FALSE(s2.has_page(0, 0));
    CHECK_FALSE(s1.has_page(0, 3));
    CHECK(s1.has_page(0, 1));
    CHECK(s2.has_page(0, 2));
    // KV came back from the right medium: page 0's disk read, page 3's pool read.
    CHECK(at::equal(pls[0].k.select(1, 0), hot));
    CHECK(at::equal(pls[1].k.select(1, 0), warm));
    // Completeness holds after the promotion: the remaining demoted pages are
    // exactly the two losers, one per bucket, plus the attended window.
    CHECK(page_indices(memory.history(0)) == (std::vector<int64_t>{1, 2, 4}));

    s2.end(0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("a kept 2-page segment recalls two per-page Payloads", "[kvmemory]") {
    auto a = cpu_alloc();
    const int64_t noted = attend_working(a, /*addr0=*/2 * PS);
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    s1.begin(0);
    // Two pages (addr 0..15, 16..31) fall in ONE segment (segment_size = 2 * PS).
    // Page 1 reranks at 0.9 and page 0 at 0.2, so best-first is the REVERSE of address
    // order and a payload list built in address order fails.
    evict_page(s1, 0, scoring_page(0.2), page_k({1, 0, 0, 0}), 1.0f);
    evict_page(s1, 16, scoring_page(0.9), page_k({0, 1, 0, 0}), 1.0f);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, /*segment_size=*/2 * PS);

    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    // The single kept 2-page segment reranks per PAGE, so a two-page capacity returns TWO
    // per-page Payloads (one page each), best-first (page 0 over page 1).
    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/2 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        recall_params(/*segment_size=*/2 * PS)
    );
    REQUIRE(pls.size() == 2);
    CHECK(pls[0].num_tokens() == PS);
    CHECK(pls[1].num_tokens() == PS);
    CHECK(pls[0].base_addr == 16);  // the higher-scoring page first
    CHECK(pls[1].base_addr == 0);

    std::filesystem::remove_all(dir);
}

TEST_CASE("recall promotes a winner's whole neighbourhood, or none of it", "[kvmemory][recall]") {
    auto a = cpu_alloc();
    std::string dir = temp_dir();
    const int64_t noted = attend_working(a, /*addr0=*/5 * PS);
    // Five pages in ONE segment (segment_size = 5 * PS). scoring_page states the target
    // as a squashed score, so in the r the rerank actually returns these are
    // 19.0, 0.111, 0.111, 4.0, 4.0: a hot page flanked by cold neighbours, then a warm
    // pair. Selection runs on those RAW scores; the neighbourhood is the unit promoted.
    auto populate = [](PagedBucket& s) {
        s.begin(0);
        const double score[5] = {0.95, 0.1, 0.1, 0.8, 0.8};
        for (int64_t p = 0; p < 5; ++p) {
            evict_page(s, p * PS, scoring_page(score[p]), page_k({1, 0, 0, 0}), 1.0f);
        }
    };
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);

    // Bar at r = 2 (tau 2/3) and radius 1, with capacity for exactly two pages.
    // Best first: p0 (19.0) wins and takes page ids {0, 1} -- p1's own 0.111 is far under
    // the bar and it is promoted purely as p0's neighbour. p3 (4.0) wins its draw next but
    // its neighbourhood {p2, p3, p4} needs three pages and only none is left, so it is
    // skipped WHOLE rather than part-promoted. p4 then fails for the same reason and p2 is
    // under the bar.
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2(dir);
    populate(s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, /*segment_size=*/5 * PS);
    RecallParams params = recall_params(/*segment_size=*/5 * PS);
    params.bar = 2.0;  // tau 2/3 -> tau / (1 - tau) = 2
    params.block_radius = 1;
    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/2 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        params
    );
    REQUIRE(pls.size() == 2);
    CHECK(pls[0].base_addr == 0);  // the winner
    CHECK(pls[1].base_addr == 16);  // rode in under the bar, as a neighbour
    // The neighbour is promoted on the WINNER's merit, so it carries the winner's score.
    CHECK(pls[0].score >= params.bar);
    CHECK(pls[1].score == pls[0].score);
    for (const auto& p : pls) {
        const int64_t base = p.base_addr;
        CHECK(base != 48);  // p3's neighbourhood did not fit, so no part of it promoted
        CHECK(base != 64);
    }

    std::filesystem::remove_all(dir);
}

TEST_CASE("recall expansion crosses a segment boundary into an unscored page", "[kvmemory][recall]") {
    auto a = cpu_alloc();
    std::string dir = temp_dir();
    const int64_t noted = attend_working(a, /*addr0=*/4 * PS);
    // Four contiguous pages (page_id 0..3), segment_size = 2 * PS: seg0 = {p0,p1},
    // seg1 = {p2,p3}. Query token TOK0 (in seg0) so the BM25 prefilter (k=1) keeps only
    // seg0: p2 and p3 are never scored and stay -inf. Rerank scores: p0=0.2, p1=0.5, so
    // the kept boundary page p1 is the winner and its neighbourhood reaches p2.
    auto populate = [](PagedBucket& s) {
        s.begin(0);
        const double score[4] = {0.2, 0.5, 0.95, 0.5};
        for (int64_t p = 0; p < 4; ++p) {
            evict_page(s, p * PS, scoring_page(score[p]), page_k({1, 0, 0, 0}), 1.0f);
        }
    };
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    RecallParams params = recall_params(/*segment_size=*/2 * PS);
    params.prefilter_k = 1;  // keep only the query token's segment (seg0)
    params.block_radius = 1;

    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2(dir);
    populate(s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, params.segment_size);
    // R=1: p1 wins and takes page ids {0, 1, 2}. p2 lost the prefilter and has no score at
    // all, yet it is promoted -- an expanded neighbour is recalled because it is context.
    // p3 is one page beyond the radius and is not.
    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/3 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        params
    );
    REQUIRE(pls.size() == 3);
    std::vector<int64_t> bases;
    for (const auto& p : pls) {
        bases.push_back(p.base_addr);
    }
    std::sort(bases.begin(), bases.end());
    CHECK(bases == std::vector<int64_t>({0, 16, 32}));
    std::filesystem::remove_all(dir);
}

TEST_CASE("recall expansion breaks across a page_id gap wider than R", "[kvmemory][recall]") {
    auto a = cpu_alloc();
    std::string dir = temp_dir();
    // The attended window is at page 12, past region B.
    const int64_t noted = attend_working(a, /*addr0=*/192);
    // Region A = page_id 0,1 (addr 0,16); region B = page_id 10,11 (addr 160,176), an
    // 8-page_id gap. segment_size = 2 * PS: seg0 = {0,1} (kept via query token), seg5 =
    // {10,11}. Scores: A0=0.5, A1=0.2 (kept boundary near the gap); B10=0.95, B11=0.5.
    auto populate = [](PagedBucket& s) {
        s.begin(0);
        evict_page(s, 0, scoring_page(0.5), page_k({1, 0, 0, 0}), 1.0f);
        evict_page(s, 16, scoring_page(0.2), page_k({1, 0, 0, 0}), 1.0f);
        evict_page(s, 160, scoring_page(0.95), page_k({1, 0, 0, 0}), 1.0f);
        evict_page(s, 176, scoring_page(0.5), page_k({1, 0, 0, 0}), 1.0f);
    };
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    RecallParams params = recall_params(/*segment_size=*/2 * PS);
    params.prefilter_k = 1;
    params.block_radius = 3;

    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2(dir);
    populate(s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    // The two regions' addresses plus the attended window's; the gap has no
    // conversation and no document.
    note_range(memory, 2 * PS, params.segment_size);
    {
        std::vector<int64_t> addr, tok;
        for (int64_t x = 160; x < noted; ++x) {
            addr.push_back(x);
            tok.push_back(TOK0 + x);
        }
        memory.note_tokens(0, addr, tok, params.segment_size);
    }
    // R=3 < the 9-page_id distance from the kept boundary p1 to B10, so p0's expansion
    // covers region A alone and never reaches region B. Distance breaks the neighbourhood
    // on its own; there is no explicit run split.
    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/5 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        params
    );
    REQUIRE(pls.size() == 2);
    CHECK(pls[0].base_addr == 0);
    CHECK(pls[1].base_addr == 16);
    for (const auto& p : pls) {
        const int64_t base = p.base_addr;
        CHECK(base != 160);  // region B is outside every winner's radius
        CHECK(base != 176);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("recall R==0 promotes each winner alone", "[kvmemory][recall]") {
    auto a = cpu_alloc();
    std::string dir = temp_dir();
    const int64_t noted = attend_working(a, /*addr0=*/4 * PS);
    // Same four pages as the cross-boundary test. With R=0 a winner takes no neighbours,
    // so only the kept seg0 pages are promoted, each on its own raw score: p0=0.5 outranks
    // p1=0.2, and the unscored seg1 pages stay in the bucket.
    auto populate = [](PagedBucket& s) {
        s.begin(0);
        const double score[4] = {0.5, 0.2, 0.95, 0.5};
        for (int64_t p = 0; p < 4; ++p) {
            evict_page(s, p * PS, scoring_page(score[p]), page_k({1, 0, 0, 0}), 1.0f);
        }
    };
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    RecallParams params = recall_params(/*segment_size=*/2 * PS);
    params.prefilter_k = 1;

    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2(dir);
    populate(s1);
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, params.segment_size);
    auto pls = memory.recall(
        0,
        query,
        /*capacity_tokens=*/3 * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        params
    );
    REQUIRE(pls.size() == 2);
    CHECK(pls[0].base_addr == 0);  // raw order, no expansion
    CHECK(pls[1].base_addr == 16);
    for (const auto& p : pls) {
        const int64_t base = p.base_addr;
        CHECK(base != 32);  // no expansion at R==0
        CHECK(base != 48);
    }
    std::filesystem::remove_all(dir);
}

// Recall stages candidate K from the bucket into the device RecallBuffer and scores it
// batched. Assert (1) at temperature 0 its selected page-id SET matches the reranker's
// own top-3 over the SAME stored K, read back through the production gather (so a
// staging permutation or a lost row shows up as a different set); (2) each promoted
// page's KV is byte-identical to what was demoted (the pool round-trips KV exactly);
// (3) at a seeded temperature > 0 two identical runs select the identical set (the
// recall RNG draw count/order is unperturbed).
TEST_CASE("staged recall matches the reranker's own ranking and round-trips KV", "[kvmemory][recall][equivalence]") {
    const int64_t N = 6, capacity_pages = 3;
    auto a = cpu_alloc();  // n_kv_heads=1, head_dim=4
    const int64_t noted = attend_working(a, /*addr0=*/N * PS);
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec({1.0, 0.5, -0.3, 0.2});

    // N pages with distinct random K/V (page p at address p*PS); fixed seed so every
    // build is byte-identical. Captures the put K/V when out-params are given.
    auto make_s1 = [&](std::vector<at::Tensor>* pk, std::vector<at::Tensor>* pv) {
        at::manual_seed(7);
        PagedBucket s = cpu_s1();
        s.begin(0);
        for (int64_t p = 0; p < N; ++p) {
            at::Tensor k = at::randn({PS, 1, 4}), v = at::randn({PS, 1, 4});
            if (pk) {
                pk->push_back(k);
                pv->push_back(v);
            }
            s.accept(0, make_page(p * PS, k, v, 1.0));
        }
        return s;
    };

    // (1) + (2): temperature 0.
    std::vector<at::Tensor> putk, putv;
    PagedBucket s1 = make_s1(&putk, &putv);
    auto buf = cpu_buffer();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);

    // Reference: the batched reranker (the recall datapath) over the pages' stored K,
    // read back through the bucket's production gather, against the same neutral
    // reference. Its top-3 is the selection staged recall must reproduce.
    std::vector<int64_t> all_pages(N);
    std::iota(all_pages.begin(), all_pages.end(), 0);
    pulsar::GatheredK cand = s1.gather_k(0, all_pages, {0});
    at::Tensor bat = AttnScoreRetriever{}
                         .rerank_score_batched(qv, cand.layers, flat_lse(), flat_log_len(), flat_positions(), PS, PS)
                         .to(at::kDouble)
                         .contiguous();
    auto ord = bat.argsort(/*dim=*/-1, /*descending=*/true);
    // The scores are in gather-row order, so the top-k names rows; `order` turns a row
    // back into the page it holds.
    std::unordered_set<int64_t> ref_set;
    for (int64_t c = 0; c < capacity_pages; ++c) {
        ref_set.insert(all_pages[cand.order[ord[c].item<int64_t>()]]);
    }

    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted);
    auto pls = memory.recall(
        0,
        query,
        capacity_pages * PS,
        qv,
        flat_lse(),
        flat_log_len(),
        flat_positions(),
        recall_params()
    );
    REQUIRE(static_cast<int64_t>(pls.size()) == capacity_pages);
    std::unordered_set<int64_t> got_set;
    for (const Payload& pl : pls) {
        const int64_t page = pl.base_addr / PS;
        got_set.insert(page);
        // Byte-identical KV: the promoted page's K/V (float32, exact) against what was
        // demoted.
        CHECK(at::equal(pl.k.select(1, 0), putk[page]));
        CHECK(at::equal(pl.v.select(1, 0), putv[page]));
    }
    CHECK(got_set == ref_set);  // staged selection == bf16 batched reranker top-k
    std::filesystem::remove_all(dir);

    // (3): seeded temperature > 0 -> two identical runs select the identical set, and a
    // different seed does not. These pages score in [0.27, 0.65] against the working
    // window, so a bar INSIDE that band puts every keep probability near 0.5: each page
    // draws a genuine coin and some survive. A bar below the whole band would keep
    // everything, and both seeds would return the same top-3 with no coin decided.
    RecallConfig tcfg = recall_cfg();
    RecallParams tparams = recall_params();
    tparams.bar = 0.72 / 0.28;
    tparams.temperature = 1.0;
    auto sel_of = [&](int64_t seed) {
        tcfg.seed = seed;
        PagedBucket s = make_s1(nullptr, nullptr);
        auto b = cpu_buffer();
        std::string d = temp_dir();
        auto disk = cpu_s2(d);
        KVMemory mem(a, b, s, disk, tcfg);
        note_range(mem, noted);
        auto p = mem.recall(0, query, capacity_pages * PS, qv, flat_lse(), flat_log_len(), flat_positions(), tparams);
        std::unordered_set<int64_t> ss;
        for (const Payload& pl : p) {
            ss.insert(pl.base_addr / PS);
        }
        std::filesystem::remove_all(d);
        return ss;
    };
    std::unordered_set<int64_t> drawn = sel_of(123);
    REQUIRE_FALSE(drawn.empty());  // the coins are real, not a bar nothing clears
    CHECK(drawn == sel_of(123));
    CHECK(drawn != sel_of(7));  // and the seed is what drives them
}

// The staging cache is demand-driven and persistent: a cycle stages only what it
// misses, and an entry stays valid for the page's whole demoted lifetime. Only two
// events end one -- the page leaving the buckets (a promotion) and the sequence
// ending. Moving between buckets is NOT one of them: a spill never touches the
// stored K, so the cached copy remains the same bytes.
TEST_CASE("the staging cache survives an s1 -> s2 spill and counts hits and promotions", "[kvmemory][recall]") {
    constexpr int64_t N = 4;
    at::manual_seed(11);  // distinct random per-page K -> a strict qk rerank order
    auto a = cpu_alloc();
    const int64_t noted = attend_working(a, /*addr0=*/N * PS);
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    std::string dir = temp_dir();
    auto s2 = cpu_s2(dir);
    s1.begin(0);
    s2.begin(0);
    for (int64_t p = 0; p < N; ++p) {
        evict_page(s1, p * PS, at::randn({PS, 1, 4}), at::randn({PS, 1, 4}), 1.0f);
    }
    KVMemory memory(a, buf, s1, s2, recall_cfg());  // qk: the cascade that stages
    note_range(memory, noted);
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec({1.0, 0.5, -0.3, 0.2});
    auto cycle = [&](RecallCounts* counts) {
        return memory.recall(
            0,
            query,
            /*capacity_tokens=*/PS,
            qv,
            flat_lse(),
            flat_log_len(),
            flat_positions(),
            recall_params(),
            /*raw_scores=*/nullptr,
            counts
        );
    };

    // Cycle 1: cold cache, every candidate is staged out of s1; one page promotes.
    RecallCounts c1;
    REQUIRE(cycle(&c1).size() == 1);
    CHECK(c1.staged_misses == N);
    CHECK(c1.staged_hits == 0);
    CHECK(c1.promoted_s1 == 1);
    CHECK(c1.promoted_s2 == 0);
    // The promoted page left the buckets, so its copy is dropped; the rest stay.
    CHECK(buf.cached_pages() == N - 1);

    // Everything still in s1 spills to s2 (a budget below one page).
    memory.end_cycle(0, /*ram_bucket_size=*/1, /*step=*/1, /*decay=*/0.0);
    REQUIRE(s1.bucket_tokens(0) == 0);
    REQUIRE(s2.bucket_tokens(0) == (N - 1) * PS);

    // Cycle 2: the same pages, now held by the other bucket. Every candidate hits,
    // so the rerank reads nothing back off disk, and the promotion comes out of s2.
    RecallCounts c2;
    REQUIRE(cycle(&c2).size() == 1);
    CHECK(c2.staged_hits == N - 1);
    CHECK(c2.staged_misses == 0);
    CHECK(c2.promoted_s1 == 0);
    CHECK(c2.promoted_s2 == 1);
    CHECK(buf.cached_pages() == N - 2);

    // Ending the sequence drops what it cached; the pool never grew for any of it.
    memory.end_seq(0);
    CHECK(buf.cached_pages() == 0);
    CHECK(buf.capacity_pages() == STAGE_PAGES);
    s2.end(0);
    std::filesystem::remove_all(dir);
}

TEST_CASE("recall_skip_fresh filters this cycle's evicted pages from the candidates", "[kvmemory][recall]") {
    // Two pages in the conversation, both in s1. Page 0 reranks at 0.9 and page 1 at
    // 0.2, so page 0 always wins on score. Marking page 0 as evicted THIS cycle makes
    // it unselectable under skip_fresh, so the colder page 1 is promoted instead. The
    // documents are identical either way -- a page's text stays in the corpus, only
    // selectability moves.
    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    // The DemotedKV an evict of page 0 emits: addresses [0, PS), token ids
    // TOK0 + addr, and the page's contiguous active-pool slot run.
    std::vector<int64_t> addr(PS), tok(PS);
    for (int64_t i = 0; i < PS; ++i) {
        addr[i] = i;
        tok[i] = TOK0 + i;
    }
    auto i64v = [](const std::vector<int64_t>& v) { return at::tensor(v, at::TensorOptions().dtype(at::kLong)); };

    for (bool skip : {true, false}) {
        auto a = cpu_alloc();
        a.allocate(0, PS);
        a.set_identity(0, 0, addr, tok);
        std::vector<int64_t> slots(PS);
        at::Tensor slot_map = a.slot_mapping(0, 0, PS).to(at::kLong);
        for (int64_t i = 0; i < PS; ++i) {
            slots[i] = slot_map[i].item<int64_t>();
        }
        // Page 0's K, so the eviction below demotes a page that reranks at 0.9.
        a.k_pool(0).reshape({-1, 1, 4}).index_copy_(0, slot_map, scoring_page(0.9));
        // The attended window is what stays behind the eviction, at page 2.
        const int64_t noted = attend_working(a, /*addr0=*/2 * PS);
        auto buf = cpu_buffer();
        auto s1 = cpu_s1();
        std::string dir = temp_dir();
        auto s2 = cpu_s2(dir);
        s1.begin(0);
        // Page 1 was demoted in an earlier cycle; page 0 is demoted in this one.
        evict_page(s1, 16, scoring_page(0.2), page_k({0, 1, 0, 0}), 1.0f);
        KVMemory memory(a, buf, s1, s2, recall_cfg(), skip);
        note_range(memory, noted, /*segment_size=*/2 * PS);
        pulsar::DemotedKV demoted;
        demoted.addr_data = i64v(addr);
        demoted.tok_data = i64v(tok);
        demoted.mass_data = at::zeros({1}, at::TensorOptions().dtype(at::kFloat));
        demoted.victim_slots = i64v(slots);
        memory.accept_evicted(0, demoted, /*step=*/1);
        REQUIRE(s1.has_page(0, 0));  // the fresh page IS held, just not selectable

        const RecallParams params = recall_params(/*segment_size=*/2 * PS);
        auto pls = memory.recall(
            0,
            query,
            /*capacity_tokens=*/PS,
            qv,
            flat_lse(),
            flat_log_len(),
            flat_positions(),
            params
        );
        REQUIRE(pls.size() == 1);
        CHECK(pls[0].base_addr == (skip ? 16 : 0));
        // end_cycle clears the fresh set, so the next cycle can select page 0.
        memory.end_cycle(0, /*ram_bucket_size=*/0, /*step=*/1, /*decay=*/0.0);
        auto next = memory.recall(
            0,
            query,
            /*capacity_tokens=*/PS,
            qv,
            flat_lse(),
            flat_log_len(),
            flat_positions(),
            params
        );
        REQUIRE(next.size() == 1);
        CHECK(next[0].base_addr == (skip ? 0 : 16));
        std::filesystem::remove_all(dir);
    }
}

TEST_CASE(
    "with no spill_dir the disk bucket is untouched and the staging pool "
    "never grows",
    "[kvmemory][recall]"
) {
    // An empty spill_dir disables s2 entirely: every demoted page stays in s1 and
    // nothing is ever written. Across many cycles the RecallBuffer -- the only GPU
    // pool the recall path allocates -- must stay at exactly its preallocated
    // capacity, since a cycle that would need more is an error, not a grow.
    constexpr int64_t N_PAGES = 10;  // > STAGE_PAGES/2, so several cycles stage a lot
    auto a = cpu_alloc();
    const int64_t noted = attend_working(a, /*addr0=*/N_PAGES * PS);
    auto buf = cpu_buffer();
    auto s1 = cpu_s1();
    auto s2 = cpu_s2("");  // disabled
    REQUIRE_FALSE(s2.enabled());
    s1.begin(0);
    s2.begin(0);  // no-op on a disabled bucket
    for (int64_t p = 0; p < N_PAGES; ++p) {
        evict_page(s1, p * PS, at::randn({PS, 1, 4}), at::randn({PS, 1, 4}), 1.0f);
    }
    KVMemory memory(a, buf, s1, s2, recall_cfg());
    note_range(memory, noted, /*segment_size=*/2 * PS);

    at::Tensor query = at::tensor(std::vector<int64_t>{TOK0}, at::TensorOptions().dtype(at::kLong));
    std::vector<at::Tensor> qv = query_vec(QUERY_AXIS);
    int64_t held = N_PAGES;
    for (int64_t cycle = 0; cycle < N_PAGES; ++cycle) {
        auto pls = memory.recall(
            0,
            query,
            /*capacity_tokens=*/PS,
            qv,
            flat_lse(),
            flat_log_len(),
            flat_positions(),
            recall_params(/*segment_size=*/2 * PS)
        );
        held -= static_cast<int64_t>(pls.size());
        memory.end_cycle(0, /*ram_bucket_size=*/0, cycle, /*decay=*/0.0);
        // The staging pool is preallocated and never reallocated.
        CHECK(buf.capacity_pages() == STAGE_PAGES);
        // Every page stays in s1; the disabled bucket holds nothing and has no seq.
        CHECK(s1.bucket_tokens(0) == held * PS);
        CHECK(s2.size() == 0);
        CHECK_FALSE(s2.has_seq(0));
        if (held == 0) {
            break;
        }
    }
    CHECK(held == 0);  // the whole conversation promoted out one page per cycle
}

namespace {

// The selection fixture: six pages straddling tau = 0.6, one of them (idx 2) never
// scored. Selection consumes r (the would-be share as a multiple of uniform), not a
// squashed score, so these are r = s / (1 - s) for s = 0.55, 0.90, --, 0.10, 0.65, 0.40.
// The bar is r_bar = tau / (1 - tau) = 1.5. Descending finite order is idx1 (9.000),
// idx4 (1.857), idx0 (1.222), idx5 (0.667), idx3 (0.111); idx1 and idx4 are above the
// bar, the other three below it -- the same partition the squashed values gave, since
// r / (1 + r) is monotone.
const std::vector<double> kStraddleScores =
    {0.55 / 0.45, 0.90 / 0.10, -std::numeric_limits<double>::infinity(), 0.10 / 0.90, 0.65 / 0.35, 0.40 / 0.60};

at::Tensor straddle_scores() {
    return at::tensor(kStraddleScores, at::TensorOptions().dtype(at::kDouble));
}

// Every page a full page (16 tokens), so capacity in pages * 16 is a page count.
std::vector<int64_t> full_pages(int64_t n) {
    return std::vector<int64_t>(n, 16);
}

// Contiguous page ids 0..n-1: the selection tests below all run at block_radius 0, where
// a winner takes no neighbours and the ids only have to be ascending.
std::vector<int64_t> seq_pids(int64_t n) {
    std::vector<int64_t> pid(n);
    std::iota(pid.begin(), pid.end(), 0);
    return pid;
}

// The bar that bars everything, as evict_bar_from_tau returns for tau >= 1.
constexpr double kInfBar = std::numeric_limits<double>::infinity();

// sigmoid((log score - log bar) / temperature), the rule spelled out independently of the
// implementation, for the expected-frequency checks.
double keep_probability(double score, double bar, double temperature) {
    return 1.0 / (1.0 + std::exp(-(std::log(score) - std::log(bar)) / temperature));
}

}  // namespace

TEST_CASE("select_recall_pages: temperature 0 is the hard bar, rng untouched", "[kvmemory][recall]") {
    at::Tensor scores = straddle_scores();
    std::mt19937_64 rng(12345);
    const std::mt19937_64 rng_before = rng;

    // tau = 0.6: only the two above-bar pages, best first. Capacity is not the limit.
    std::vector<int64_t> sel = pulsar::select_recall_pages(
        scores,
        full_pages(6),
        seq_pids(6),
        /*block_radius=*/0,
        /*capacity_tokens=*/6 * 16,
        /*bar=*/0.6 / 0.4,
        /*temperature=*/0.0,
        rng
    );
    CHECK(sel == std::vector<int64_t>{1, 4});
    CHECK(rng == rng_before);  // temperature 0 must not draw from the rng

    // The same selection is what today's deterministic top-k produces: a descending walk
    // over the scores the rerank kept (the sub-bar pages dropped to -inf) stops at the
    // first -inf, so the two rules agree page for page.
    std::vector<double> thresholded = kStraddleScores;
    for (double& r : thresholded) {  // the bar in r's units: tau / (1 - tau) at tau 0.6
        if (r < 0.6 / 0.4) {
            r = -std::numeric_limits<double>::infinity();
        }
    }
    std::vector<int64_t> deterministic;  // descending argsort, stop at the first -inf
    {
        std::vector<int64_t> order(thresholded.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return thresholded[a] > thresholded[b];
        });
        for (int64_t i : order) {
            if (!std::isfinite(thresholded[i])) {
                break;
            }
            deterministic.push_back(i);
        }
    }
    CHECK(sel == deterministic);

    // tau = 0 puts the bar at -inf: every scored page passes, the never-scored one still
    // does not. tau = 1 puts it at +inf: nothing passes. Neither produces a NaN keep
    // probability, and neither draws.
    CHECK(
        pulsar::select_recall_pages(
            scores,
            full_pages(6),
            seq_pids(6),
            /*block_radius=*/0,
            6 * 16,
            /*bar=*/0.0,
            /*temperature=*/0.0,
            rng
        ) == std::vector<int64_t>{1, 4, 0, 5, 3}
    );
    CHECK(
        pulsar::select_recall_pages(
            scores,
            full_pages(6),
            seq_pids(6),
            /*block_radius=*/0,
            6 * 16,
            /*bar=*/kInfBar,
            /*temperature=*/0.0,
            rng
        )
            .empty()
    );
    CHECK(rng == rng_before);
}

TEST_CASE("select_recall_pages: the bar's ends never draw or produce a NaN", "[kvmemory][recall]") {
    // tau at either end is resolved by comparison at ANY temperature (log r_bar is
    // infinite there, so a subtraction would be inf - inf), and so is r at 0.
    // r values, not squashed scores: 0 (score 0), 1e12 (score ~1, and finite because
    // non-finite marks a page selection may not take), 1.0 (score 0.5).
    at::Tensor ends = at::tensor(std::vector<double>{0.0, 1e12, 1.0}, at::TensorOptions().dtype(at::kDouble));
    std::mt19937_64 rng(7);
    const std::mt19937_64 rng_before = rng;
    CHECK(
        pulsar::select_recall_pages(
            ends,
            full_pages(3),
            seq_pids(3),
            /*block_radius=*/0,
            3 * 16,
            /*bar=*/0.0,
            /*temperature=*/2.0,
            rng
        ) == std::vector<int64_t>{1, 2, 0}
    );
    CHECK(
        pulsar::select_recall_pages(
            ends,
            full_pages(3),
            seq_pids(3),
            /*block_radius=*/0,
            3 * 16,
            /*bar=*/kInfBar,
            /*temperature=*/2.0,
            rng
        )
            .empty()
    );
    CHECK(rng == rng_before);  // a degenerate bar costs no draw

    // With the bar strictly inside (0, 1), a score of 1 is certain and a score of 0 is
    // impossible, both without a draw; only the 0.5 page flips a coin.
    for (int64_t i = 0; i < 50; ++i) {
        std::vector<int64_t> sel = pulsar::select_recall_pages(
            ends,
            full_pages(3),
            seq_pids(3),
            /*block_radius=*/0,
            3 * 16,
            /*bar=*/0.6 / 0.4,
            /*temperature=*/1.0,
            rng
        );
        REQUIRE_FALSE(sel.empty());
        CHECK(sel[0] == 1);  // score 1 always promotes, and ranks first
        for (int64_t idx : sel) {
            CHECK(idx != 0);  // score 0 never promotes
        }
    }
}

TEST_CASE("select_recall_pages: temperature > 0 keeps each page at its own rate", "[kvmemory][recall]") {
    // Every page is drawn independently, so over many seeded draws each page's observed
    // frequency must land on sigmoid((logit(score) - logit(tau)) / temperature). Capacity
    // is wide open so nothing is cut short.
    at::Tensor scores = straddle_scores();
    const double tau = 0.6, temperature = 0.5;
    std::mt19937_64 rng(2024);
    const int64_t N = 20000;
    std::vector<int64_t> tally(6, 0);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t idx : pulsar::select_recall_pages(
                 scores,
                 full_pages(6),
                 seq_pids(6),
                 /*block_radius=*/0,
                 6 * 16,
                 tau / (1.0 - tau),
                 temperature,
                 rng
             )) {
            tally[idx] += 1;
        }
    }
    for (int64_t idx = 0; idx < 6; ++idx) {
        if (!std::isfinite(kStraddleScores[idx])) {
            CHECK(tally[idx] == 0);  // a never-scored page is never a candidate
            continue;
        }
        const double expected = keep_probability(kStraddleScores[idx], tau / (1.0 - tau), temperature);
        const double observed = static_cast<double>(tally[idx]) / static_cast<double>(N);
        // 20000 draws: the standard error is at most 0.0035, so 0.02 is ~6 sigma.
        CHECK(std::abs(observed - expected) < 0.02);
    }
    // Pages on both sides of the bar are reached: the sub-bar page at 0.55 is sometimes
    // promoted and the above-bar page at 0.65 is sometimes skipped.
    CHECK(tally[0] > 0);
    CHECK(tally[4] < N);
}

TEST_CASE("select_recall_pages: a fixed seed reproduces the same draws", "[kvmemory][recall]") {
    at::Tensor scores = straddle_scores();
    auto run = [&](uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::vector<std::vector<int64_t>> cycles;
        for (int64_t i = 0; i < 20; ++i) {
            cycles.push_back(
                pulsar::select_recall_pages(
                    scores,
                    full_pages(6),
                    seq_pids(6),
                    /*block_radius=*/0,
                    6 * 16,
                    /*bar=*/0.6 / 0.4,
                    /*temperature=*/1.0,
                    rng
                )
            );
        }
        return cycles;
    };
    CHECK(run(99) == run(99));  // same seed, same sequence of selections
    CHECK(run(99) != run(1000000));  // and the seed actually drives them
}

TEST_CASE("select_recall_pages: capacity is a token budget and is never exceeded", "[kvmemory][recall]") {
    at::Tensor scores = straddle_scores();
    std::mt19937_64 rng(5);
    // Page tokens: page 4 is a short TAIL page (5 tokens), the rest are full.
    std::vector<int64_t> page_tokens = {16, 16, 16, 16, 5, 16};

    // Descending order is idx1 (16), idx4 (5), idx0 (16), idx5 (16), idx3 (16). At tau 0
    // every page passes its draw, so a 37-token capacity takes idx1 + idx4 + idx0 exactly
    // and stops at idx5, which would need 53.
    std::vector<int64_t> sel = pulsar::select_recall_pages(
        scores,
        page_tokens,
        seq_pids(6),
        /*block_radius=*/0,
        /*capacity_tokens=*/37,
        /*bar=*/0.0,
        /*temperature=*/0.0,
        rng
    );
    CHECK(sel == std::vector<int64_t>{1, 4, 0});

    // A capacity of 21 fits the top page plus the short tail page but no second full
    // page: the stop is on TOKENS, so two pages fit where a page count of 21 / 16 == 1
    // would take one.
    CHECK(
        pulsar::select_recall_pages(
            scores,
            page_tokens,
            seq_pids(6),
            /*block_radius=*/0,
            /*capacity_tokens=*/21,
            /*bar=*/0.0,
            /*temperature=*/0.0,
            rng
        ) == std::vector<int64_t>{1, 4}
    );
    // Below the first candidate's own size nothing fits.
    CHECK(
        pulsar::select_recall_pages(
            scores,
            page_tokens,
            seq_pids(6),
            /*block_radius=*/0,
            /*capacity_tokens=*/4,
            /*bar=*/0.0,
            /*temperature=*/0.0,
            rng
        )
            .empty()
    );
    CHECK(
        pulsar::select_recall_pages(
            scores,
            page_tokens,
            seq_pids(6),
            /*block_radius=*/0,
            /*capacity_tokens=*/0,
            /*bar=*/0.0,
            /*temperature=*/0.0,
            rng
        )
            .empty()
    );

    // Under sampling the promoted tokens still never exceed the capacity, and the
    // selection is always in descending score.
    for (int64_t i = 0; i < 500; ++i) {
        std::vector<int64_t> s = pulsar::select_recall_pages(
            scores,
            page_tokens,
            seq_pids(6),
            /*block_radius=*/0,
            /*capacity_tokens=*/37,
            /*bar=*/0.6 / 0.4,
            /*temperature=*/1.0,
            rng
        );
        int64_t tokens = 0;
        for (int64_t idx : s) {
            tokens += page_tokens[idx];
        }
        CHECK(tokens <= 37);
        for (size_t j = 1; j < s.size(); ++j) {
            CHECK(kStraddleScores[s[j - 1]] >= kStraddleScores[s[j]]);
        }
    }
}

TEST_CASE("select_recall_pages: a higher-scoring page is skipped only by its own coin", "[kvmemory][recall]") {
    // Capacity for ONE page and two candidates. The lower-scoring page can only be
    // promoted when the higher-scoring one drew a skip, never by claiming the capacity
    // first, so the pair (top skipped, second promoted) is the only way idx1 appears.
    at::Tensor scores = at::tensor(std::vector<double>{0.80, 0.62}, at::TensorOptions().dtype(at::kDouble));
    std::mt19937_64 rng(31337);
    int64_t top = 0, second = 0;
    const int64_t N = 4000;
    for (int64_t i = 0; i < N; ++i) {
        std::vector<int64_t> sel = pulsar::select_recall_pages(
            scores,
            full_pages(2),
            seq_pids(2),
            /*block_radius=*/0,
            /*capacity_tokens=*/16,
            /*bar=*/0.6 / 0.4,
            /*temperature=*/1.0,
            rng
        );
        REQUIRE(sel.size() <= 1);  // one page of capacity
        if (sel.empty()) {
            continue;
        }
        if (sel[0] == 0) {
            ++top;
        }
        if (sel[0] == 1) {
            ++second;
        }
    }
    // The top page is promoted at its own rate, unaffected by the page below it.
    const double p_top = keep_probability(0.80, 0.6 / 0.4, 1.0);
    CHECK(std::abs(static_cast<double>(top) / static_cast<double>(N) - p_top) < 0.03);
    // The second page appears only on the top page's skips, at (1 - p_top) * p_second.
    const double p_second = keep_probability(0.62, 0.6 / 0.4, 1.0);
    CHECK(std::abs(static_cast<double>(second) / static_cast<double>(N) - (1.0 - p_top) * p_second) < 0.03);
}

TEST_CASE("select_recall_pages: an expanded neighbour carries its winner's score", "[kvmemory][recall]") {
    // Five contiguous pages. Page 2 is the only one above the bar; pages 1 and 3 were
    // never scored (-inf) and can enter only as its neighbours, on its merit.
    const double neg_inf = -std::numeric_limits<double>::infinity();
    at::Tensor scores = at::tensor(
        std::vector<double>{0.5, neg_inf, 3.0, neg_inf, neg_inf},
        at::TensorOptions().dtype(at::kDouble)
    );
    std::mt19937_64 rng(4242);
    std::vector<double> sel_score;

    std::vector<int64_t> sel = pulsar::select_recall_pages(
        scores,
        full_pages(5),
        seq_pids(5),
        /*block_radius=*/1,
        /*capacity_tokens=*/5 * 16,
        /*bar=*/1.0,
        /*temperature=*/0.0,
        rng,
        &sel_score
    );
    CHECK(sel == std::vector<int64_t>{1, 2, 3});
    CHECK(sel_score == std::vector<double>{3.0, 3.0, 3.0});

    // With no neighbourhood every promoted page carries its OWN score, in the walk's
    // descending order, and the out-param stays parallel to the returned rows.
    sel = pulsar::select_recall_pages(
        scores,
        full_pages(5),
        seq_pids(5),
        /*block_radius=*/0,
        /*capacity_tokens=*/5 * 16,
        /*bar=*/0.0,
        /*temperature=*/0.0,
        rng,
        &sel_score
    );
    CHECK(sel == std::vector<int64_t>{2, 0});
    CHECK(sel_score == std::vector<double>{3.0, 0.5});

    // A call that promotes nothing clears the out-param rather than leaving the last
    // cycle's scores behind it.
    sel = pulsar::select_recall_pages(
        scores,
        full_pages(5),
        seq_pids(5),
        /*block_radius=*/0,
        /*capacity_tokens=*/0,
        /*bar=*/0.0,
        /*temperature=*/0.0,
        rng,
        &sel_score
    );
    CHECK(sel.empty());
    CHECK(sel_score.empty());
}
