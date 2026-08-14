#include "pulsar/runtime/kv/recall.hpp"

#include <ATen/ATen.h>

// torch's logging header defines a CHECK macro; drop it so Catch2's CHECK wins.
#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

// RecallBuffer is a persistent cache of candidate pages' relevance-layer K in a
// preallocated pool. A cycle declares its scored set, only the misses are filled,
// entries survive to later cycles, the least-recently-used one evicts when the pool
// fills, and the reranker gets its candidates gathered back into scored order. A
// cycle larger than the pool is refused, never clamped. The cascade's stages (BM25
// prefilter, qk rerank) are plain value types tested directly. The routing across
// the buffer and the bucket chain lives in the KVMemory (test_kv_memory).

namespace {

using pulsar::AttnScoreRetriever;
using pulsar::CascadeRetriever;
using pulsar::RecallBuffer;
using pulsar::RetrievalDoc;

constexpr int64_t PAGE = 16;

at::Tensor i64(std::vector<int64_t> v) {
    return at::tensor(v, at::TensorOptions().dtype(at::kLong));
}

// One page's stored K for one relevance layer, [PAGE, 1, 4], distinct per (page id,
// layer, token offset) so a misplaced row is visible.
at::Tensor page_src(int64_t page_id, int64_t li) {
    at::Tensor t = at::empty({PAGE, 1, 4}, at::TensorOptions().dtype(at::kFloat));
    for (int64_t o = 0; o < PAGE; ++o) {
        for (int64_t j = 0; j < 4; ++j) {
            t[o][0][j] = static_cast<float>(page_id * 1000 + li * 100 + o) + 0.25f * static_cast<float>(j);
        }
    }
    return t;
}

// The pages' layer-li K stacked as [n, PAGE, 1, 4] -- the shape a bucket's batched
// gather hands the buffer, and the reference candidates() must reproduce.
at::Tensor stack_src(const std::vector<int64_t>& page_ids, int64_t li) {
    std::vector<at::Tensor> v;
    v.reserve(page_ids.size());
    for (int64_t p : page_ids) {
        v.push_back(page_src(p, li));
    }
    return at::stack(v, 0);
}

// One recall cycle over `ids` (the scored set, in scored order): bind it, fill
// exactly the rows that missed, and return those rows.
std::vector<int64_t> run_cycle(RecallBuffer& buf, int64_t n_layers, int64_t seq, const std::vector<int64_t>& ids) {
    std::vector<int64_t> miss = buf.begin_cycle(seq, ids);
    if (miss.empty()) {
        return miss;
    }
    std::vector<int64_t> miss_ids;
    miss_ids.reserve(miss.size());
    for (int64_t r : miss) {
        miss_ids.push_back(ids[r]);
    }
    for (int64_t li = 0; li < n_layers; ++li) {
        buf.fill(li, miss, stack_src(miss_ids, li));
    }
    return miss;
}

}  // namespace

TEST_CASE("RecallBuffer keeps candidates across cycles and gathers scored order", "[recall]") {
    // 2 relevance layers, 1 kv head, head_dim 4, capacity 4 pages.
    constexpr int64_t NL = 2;
    RecallBuffer buf(
        NL,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PAGE,
        std::string("float32"),
        /*cycle_pages=*/4,
        /*max_sessions=*/1
    );
    CHECK(buf.capacity_pages() == 4);
    CHECK(buf.capacity_tokens() == 4 * PAGE);
    CHECK(buf.cached_pages() == 0);

    // Every row of every layer must be the page the scored set named at that row --
    // the property the old contiguous fill got for free and the gather must restore.
    auto check_order = [&](const std::vector<int64_t>& ids) {
        for (int64_t li = 0; li < NL; ++li) {
            at::Tensor c = buf.candidates(li);
            REQUIRE(c.sizes() == at::IntArrayRef({static_cast<int64_t>(ids.size()), PAGE, 1, 4}));
            CHECK(at::equal(c, stack_src(ids, li)));
        }
    };

    // Cold cache: every candidate misses and is staged in scored order.
    CHECK(run_cycle(buf, NL, 0, {10, 11, 12}) == std::vector<int64_t>{0, 1, 2});
    CHECK(buf.staged_pages() == 3);
    CHECK(buf.cached_pages() == 3);
    check_order({10, 11, 12});

    // The same pages in a different scored order: no transfer, and the gather has to
    // permute them. A wrong index here is a silently wrong rerank score.
    CHECK(run_cycle(buf, NL, 0, {12, 10, 11}).empty());
    check_order({12, 10, 11});
    // Restaging a resident row would defeat the cache and is refused.
    CHECK_THROWS(buf.fill(0, {0}, stack_src({12}, 0)));

    // A subset plus a newcomer: only the newcomer's row is staged, and it lands
    // between two cached rows.
    CHECK(run_cycle(buf, NL, 0, {11, 13, 10}) == std::vector<int64_t>{1});
    check_order({11, 13, 10});
    CHECK(buf.cached_pages() == 4);
    CHECK(buf.capacity_pages() == 4);  // preallocated, never grows

    // A shorter cycle leaves the rest cached.
    CHECK(run_cycle(buf, NL, 0, {13}).empty());
    CHECK(buf.staged_pages() == 1);
    check_order({13});
    CHECK(buf.cached_pages() == 4);
}

TEST_CASE("RecallBuffer evicts the least-recently-used entry when the pool fills", "[recall]") {
    RecallBuffer buf(
        /*n_rel_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PAGE,
        std::string("float32"),
        /*cycle_pages=*/3,
        /*max_sessions=*/1
    );
    auto cycle = [&](const std::vector<int64_t>& ids) { return run_cycle(buf, /*n_layers=*/1, /*seq=*/0, ids); };
    cycle({0, 1, 2});  // fills the pool
    CHECK(buf.cached_pages() == 3);
    // Touching 0 and 2 leaves 1 as the least-recently-used entry.
    CHECK(cycle({0, 2}).empty());
    CHECK(cycle({3}) == std::vector<int64_t>{0});  // 3 misses; 1 is the victim
    CHECK(buf.cached_pages() == 3);  // bounded by the pool
    CHECK(cycle({0, 2, 3}).empty());  // the touched entries survived
    CHECK(cycle({1}) == std::vector<int64_t>{0});  // 1 is the one that was dropped
}

TEST_CASE("RecallBuffer drops a promoted page and keys entries per sequence", "[recall]") {
    RecallBuffer buf(
        /*n_rel_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PAGE,
        std::string("float32"),
        /*cycle_pages=*/4,
        /*max_sessions=*/1
    );
    auto cycle = [&](int64_t seq, const std::vector<int64_t>& ids) { return run_cycle(buf, /*n_layers=*/1, seq, ids); };
    cycle(0, {5, 6});
    CHECK(buf.cached_pages() == 2);
    // A promotion takes the page out of the buckets -- the only event that
    // invalidates a cached copy.
    buf.drop(/*seq=*/0, /*page_id=*/5);
    CHECK(buf.cached_pages() == 1);
    CHECK(cycle(0, {5, 6}) == std::vector<int64_t>{0});  // 5 restaged, 6 still cached
    // page_id is per sequence, so another sequence's page 5 is a separate entry.
    CHECK(cycle(1, {5}) == std::vector<int64_t>{0});
    CHECK(buf.cached_pages() == 3);
    buf.end_seq(0);
    CHECK(buf.cached_pages() == 1);
    CHECK(cycle(1, {5}).empty());  // seq 1's entry survived seq 0's end
}

TEST_CASE("RecallBuffer refuses a cycle larger than ONE sequence's budget", "[recall]") {
    RecallBuffer buf(
        /*n_rel_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PAGE,
        std::string("float32"),
        /*cycle_pages=*/2,
        /*max_sessions=*/1
    );
    // The whole scored set must be simultaneously resident, so the budget is a hard
    // floor at the worst-case scored set. Exactly full is fine.
    CHECK(run_cycle(buf, /*n_layers=*/1, /*seq=*/0, {0, 1}).size() == 2);
    CHECK(buf.staged_pages() == 2);
    // One page over is a hard error, never a silent clamp and never a grow.
    CHECK_THROWS(buf.begin_cycle(/*seq=*/0, {0, 1, 2}));
    CHECK(buf.capacity_pages() == 2);
}

// The pool is max_sessions of the per-sequence budget, the same guarantee ActiveBuffer
// gives with max_sessions whole windows: every live session can hold a full cycle at
// once, so no session's staging is evicted by another's. With a pool sized for one
// session these two would thrash each other every cycle.
TEST_CASE("RecallBuffer sizes its pool for every session's cycle at once", "[recall]") {
    constexpr int64_t BUDGET = 3, SESSIONS = 2;
    RecallBuffer buf(
        /*n_rel_layers=*/1,
        /*n_kv_heads=*/1,
        /*head_dim=*/4,
        PAGE,
        std::string("float32"),
        BUDGET,
        SESSIONS
    );
    CHECK(buf.capacity_pages() == BUDGET * SESSIONS);

    // Each session stages a FULL budget's worth. Both sets stay cached.
    CHECK(run_cycle(buf, /*n_layers=*/1, /*seq=*/0, {0, 1, 2}).size() == 3);
    CHECK(run_cycle(buf, /*n_layers=*/1, /*seq=*/1, {0, 1, 2}).size() == 3);
    CHECK(buf.cached_pages() == BUDGET * SESSIONS);

    // Re-running either session's cycle is now an all-hit: the other session's full
    // cycle did not displace it.
    CHECK(run_cycle(buf, /*n_layers=*/1, /*seq=*/0, {0, 1, 2}).empty());
    CHECK(run_cycle(buf, /*n_layers=*/1, /*seq=*/1, {0, 1, 2}).empty());

    // A cycle is still bounded by ONE sequence's budget, not by the whole pool, so a
    // session cannot eat into another's share.
    CHECK_THROWS(buf.begin_cycle(/*seq=*/0, {0, 1, 2, 3}));
}

TEST_CASE("cascade prefilter ranks fixed-boundary segments on raw BM25", "[recall]") {
    // Seven address-ordered segments; only segment 4 carries the rare query token, so
    // raw BM25 spikes there and is ~0 elsewhere. The hit sits OUTSIDE the first
    // prefilter_k segments, so a prefilter that ignored BM25 and took the first k
    // would miss it.
    std::vector<RetrievalDoc> segs;
    for (int i = 0; i < 7; ++i) {
        std::vector<int64_t> d(PAGE, 5);
        if (i == 4) {
            d[0] = 100;
        }
        segs.push_back(RetrievalDoc{i64(d), {}});
    }
    RetrievalDoc query{i64({100}), {}};
    CascadeRetriever casc;

    // Segments are fixed-boundary documents scored as they stand: the hit is kept first
    // and the other two slots are BM25 ties filled in index order. Contiguity around the
    // hit is not the prefilter's job -- selection expands over page_id neighbourhoods,
    // which reach across segment boundaries.
    auto kept = casc.prefilter(query, segs, /*prefilter_k=*/3);
    REQUIRE(kept.size() == 3);
    CHECK(kept[0] == 4);  // the only segment with a nonzero score, ranked first
    std::sort(kept.begin(), kept.end());
    CHECK(kept == std::vector<int64_t>({0, 1, 4}));
}

namespace {

// A [n, 1, head_dim] single-head vec tensor from per-key rows (recall RetrievalDoc
// vecs: query Q [T, H, d], segment/working K [len, H, d]).
at::Tensor vecs1(const std::vector<std::vector<float>>& rows) {
    const int64_t n = static_cast<int64_t>(rows.size());
    const int64_t d = static_cast<int64_t>(rows[0].size());
    at::Tensor t = at::empty({n, 1, d}, at::TensorOptions().dtype(at::kFloat));
    auto a = t.accessor<float, 3>();
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            a[i][0][j] = rows[i][j];
        }
    }
    return t;
}

// Query rows all at position 0, so the rerank's per-row rotation is the identity and the
// fixture's raw q.k is the score. The reference below computes the unrotated dot product,
// so it is only comparable against the batched kernel under these positions and with the
// candidate scored where it is stored.
at::Tensor zero_positions(int64_t n_rows) {
    return at::zeros({n_rows}, at::TensorOptions().dtype(at::kLong));
}
constexpr int64_t UNROTATED = 0;  // stored == scored: the keys are used as they are

// Independent scalar reference for rerank_score_batched: the same mass-space attention
// score computed one page at a time in fp32. Test-only -- the engine has no per-page
// scorer -- and the only check on the batched kernel that is not the kernel itself.
at::Tensor qk_reference(
    const AttnScoreRetriever& scorer,
    const RetrievalDoc& query,
    const std::vector<RetrievalDoc>& pages,
    const std::vector<at::Tensor>& lse,
    const at::Tensor& log_attended_len
) {
    const int64_t n_docs = static_cast<int64_t>(pages.size());
    const int64_t n_layers = static_cast<int64_t>(query.vecs.size());
    std::vector<double> best(n_docs, 0.0);
    for (int64_t li = 0; li < n_layers; ++li) {
        at::Tensor q = query.vecs[li].to(at::kFloat).cpu();  // [T, H_q, d]
        const int64_t n_query_heads = q.size(1), d = q.size(2);
        const double scale = scorer.scale > 0.0 ? scorer.scale : 1.0 / std::sqrt(static_cast<double>(d));
        at::Tensor query_block = q.transpose(0, 1).contiguous();  // [H_q, T, d]
        // [H_q, T]: the row's captured denominator, less its log key count.
        at::Tensor ref = lse[li].to(at::kFloat).cpu().transpose(0, 1) -
            log_attended_len.to(at::kFloat).cpu().unsqueeze(0);
        for (int64_t s = 0; s < n_docs; ++s) {
            at::Tensor k = pages[s].vecs[li].to(at::kFloat).cpu();  // [ps, H_kv, d]
            if (k.size(0) == 0) {
                continue;
            }
            at::Tensor block = k.transpose(0, 1).repeat_interleave(n_query_heads / k.size(1),
                                                                   0);  // [H_q, ps, d]
            at::Tensor logits = at::bmm(query_block, block.transpose(1, 2)) * scale;  // [H_q, T, ps]
            // [H_q, T, ps]: mean over query tokens, then mean over heads, then max
            // over slots -- the eviction density's order.
            at::Tensor log_r = logits - ref.unsqueeze(-1);
            at::Tensor per_slot_head = at::logsumexp(log_r, /*dim=*/1) - std::log(static_cast<double>(log_r.size(1)));
            at::Tensor per_slot = at::logsumexp(per_slot_head, /*dim=*/0) -
                std::log(static_cast<double>(per_slot_head.size(0)));  // [ps]
            // r itself, not a squashed score: the squash belongs at the reporting
            // boundary so selection works in the density's units.
            const double per_layer = at::exp(std::get<0>(per_slot.max(/*dim=*/0))).item<double>();
            best[s] = std::max(best[s], per_layer);
        }
    }
    return at::tensor(best, at::TensorOptions().dtype(at::kDouble));
}

// Score `pages` (all the same key count, so they stack) with the production batched
// reranker, returning CPU fp64 [n_pages].
at::Tensor score_pages(
    const AttnScoreRetriever& scorer,
    const RetrievalDoc& query,
    const std::vector<RetrievalDoc>& pages,
    const std::vector<at::Tensor>& lse,
    const at::Tensor& log_attended_len
) {
    std::vector<at::Tensor> cand_k;
    cand_k.reserve(query.vecs.size());
    for (size_t li = 0; li < query.vecs.size(); ++li) {
        std::vector<at::Tensor> parts;
        parts.reserve(pages.size());
        for (const RetrievalDoc& p : pages) {
            parts.push_back(p.vecs[li]);
        }
        cand_k.push_back(at::stack(parts, 0));  // [n_docs, ps, H_kv, d]
    }
    return scorer
        .rerank_score_batched(
            query.vecs,
            cand_k,
            lse,
            log_attended_len,
            zero_positions(log_attended_len.numel()),
            UNROTATED,
            UNROTATED
        )
        .to(at::kCPU, at::kDouble)
        .contiguous();
}

// One layer, one query token: reference `ref`, so a page scores sigmoid(logit - ref).
std::vector<at::Tensor> ref1(double ref) {
    return {at::full({1, 1}, ref, at::TensorOptions().dtype(at::kFloat))};
}
at::Tensor loglen1() {
    return at::zeros({1}, at::TensorOptions().dtype(at::kFloat));
}

}  // namespace

TEST_CASE("qk rerank refuses to score without a captured reference", "[recall]") {
    // The captured denominator is what a candidate's best key is scored against.
    // Without one there is no reference, so the reranker refuses rather than scoring on
    // some other scale that the same recall_tau would then be read against.
    RetrievalDoc query{i64({0}), {vecs1({{4, 0, 0, 0}})}};
    RetrievalDoc page{i64({1, 1}), {vecs1({{3, 0, 0, 0}, {0, 3, 0, 0}})}};
    std::vector<RetrievalDoc> kept{page};
    CHECK_THROWS(score_pages(AttnScoreRetriever{}, query, kept, {}, loglen1()));
    CHECK_THROWS(score_pages(AttnScoreRetriever{}, query, kept, ref1(0.0), at::Tensor{}));
}

TEST_CASE("qk rerank returns the would-be share r, unsquashed", "[recall]") {
    // One layer, one query token, one head, one key per page (scale 1/sqrt(4) = 0.5).
    // A reference of 0 leaves r at exp(page_best_logit): a page whose single key is
    // [a, 0, 0, 0] has page_best_logit = 0.5 * 4a = 2a, so r = exp(2a). Parity -- a
    // uniform share, r = 1 -- is at a = 0.
    RetrievalDoc query{i64({0}), {vecs1({{4, 0, 0, 0}})}};
    // Ascending in a, so ascending in r: four below parity, parity, four above.
    const std::vector<float> a{-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
    std::vector<RetrievalDoc> kept;
    for (size_t i = 0; i < a.size(); ++i) {
        kept.push_back(RetrievalDoc{i64({static_cast<int64_t>(i)}), {vecs1({{a[i], 0, 0, 0}})}});
    }

    at::Tensor sc = score_pages(AttnScoreRetriever{}, query, kept, ref1(0.0), loglen1());
    auto r = sc.accessor<double, 1>();
    // r is unbounded and spans five orders of magnitude here, so the tolerance is
    // RELATIVE: an absolute bound tight enough at 0.0025 is meaningless at 2981.
    for (size_t i = 0; i < a.size(); ++i) {
        const double want = std::exp(2.0 * static_cast<double>(a[i]));
        CHECK(std::abs(r[i] - want) / want < 1e-4);
        CHECK(r[i] > 0.0);  // a share, so non-negative and unbounded above
    }
    // Parity is r == 1, not a midpoint: the squash to r / (1 + r) that puts it at 0.5
    // happens at the reporting boundary, after selection.
    CHECK(std::abs(r[4] - 1.0) < 1e-5);
    // exp(2a) at the fixture's a, hand-evaluated.
    CHECK(std::abs(r[0] - 0.00247875218) / 0.00247875218 < 1e-4);  // a = -3
    CHECK(std::abs(r[2] - 0.13533528324) / 0.13533528324 < 1e-4);  // a = -1
    CHECK(std::abs(r[6] - 7.38905609893) / 7.38905609893 < 1e-4);  // a =  1
    CHECK(std::abs(r[8] - 2980.9579870417) / 2980.9579870417 < 1e-4);  // a =  4
    // Strictly monotone on BOTH sides of parity: r is unbounded above, so the
    // above-parity pages separate instead of compressing toward a ceiling.
    for (size_t i = 1; i < a.size(); ++i) {
        CHECK(r[i] > r[i - 1]);
    }
    for (size_t i = 5; i < a.size(); ++i) {
        CHECK(r[i] > 1.0);
    }
    // The rerank applies no keep-bar: the four below-parity pages come back with their
    // own r rather than dropped, which is what lets a temperature > 0 draw reach them.
}

TEST_CASE("qk rerank states r against the configured reference length", "[recall]") {
    // Same fixture as above: r = exp(2a) with the per-row attended length (loglen1 is
    // log 1). A reference length replaces that per-row length everywhere, so every
    // page's r is multiplied by it and parity moves to the key whose share is
    // 1 / reference.
    RetrievalDoc query{i64({0}), {vecs1({{4, 0, 0, 0}})}};
    const std::vector<float> a{-3.0f, -1.0f, 0.0f, 1.0f};
    std::vector<RetrievalDoc> kept;
    for (size_t i = 0; i < a.size(); ++i) {
        kept.push_back(RetrievalDoc{i64({static_cast<int64_t>(i)}), {vecs1({{a[i], 0, 0, 0}})}});
    }

    AttnScoreRetriever unscaled;
    at::Tensor base = score_pages(unscaled, query, kept, ref1(0.0), loglen1());

    const int64_t reference = 512;
    AttnScoreRetriever scaled;
    scaled.mass_reference_length = reference;
    at::Tensor got = score_pages(scaled, query, kept, ref1(0.0), loglen1());

    auto b = base.accessor<double, 1>();
    auto g = got.accessor<double, 1>();
    for (size_t i = 0; i < a.size(); ++i) {
        const double want = b[i] * static_cast<double>(reference);
        CHECK(std::abs(g[i] - want) / want < 1e-4);
    }
    // A reference length only rescales, so the ranking selection consumes is unmoved.
    for (size_t i = 1; i < a.size(); ++i) {
        CHECK(g[i] > g[i - 1]);
    }
    // The per-row attended length is not read at all once a reference is set: a row
    // claiming a different length scores identically.
    at::Tensor other_len = at::full({1}, 7.0, at::TensorOptions().dtype(at::kFloat));
    at::Tensor same = score_pages(scaled, query, kept, ref1(0.0), other_len);
    auto s = same.accessor<double, 1>();
    for (size_t i = 0; i < a.size(); ++i) {
        CHECK(std::abs(s[i] - g[i]) / g[i] < 1e-9);
    }
}

namespace {

// Random multi-layer query/page/working docs on `dev`, GQA (H_q = group * H_kv).
// Candidate pages all have the same length ps so the batched path can stack them.
struct QkFixture {
    RetrievalDoc query;
    std::vector<RetrievalDoc> pages;
    std::vector<at::Tensor> lse;  // per layer [T, H_q], the captured denominator
    at::Tensor log_attended_len;  // [T]
    std::vector<at::Tensor> cand_k;  // per layer [n_docs, ps, H_kv, d], the batched stack
};

// ref_level is the reference each page's best-key logit is scored against
// (lse - log attended_len), so a page scores sigmoid(logit - ref_level). It must sit
// ABOVE the pages' logits: a reference below them saturates every score at ~1 and the
// batched and reference paths then agree for free, asserting nothing.
QkFixture make_qk_fixture(
    int64_t n_layers,
    int64_t n_docs,
    int64_t T,
    int64_t ps,
    int64_t H_q,
    int64_t H_kv,
    int64_t d,
    at::Device dev,
    double ref_level
) {
    auto opt = at::TensorOptions().dtype(at::kFloat).device(dev);
    QkFixture f;
    f.pages.resize(n_docs);
    for (int64_t li = 0; li < n_layers; ++li) {
        f.query.vecs.push_back(at::randn({T, H_q, d}, opt));
        std::vector<at::Tensor> parts(n_docs);
        for (int64_t s = 0; s < n_docs; ++s) {
            at::Tensor k = at::randn({ps, H_kv, d}, opt);
            f.pages[s].vecs.push_back(k);
            parts[s] = k;
        }
        f.cand_k.push_back(at::stack(parts, 0));  // [n_docs, ps, H_kv, d]
        f.lse.push_back(at::full({T, H_q}, ref_level, opt));
    }
    f.log_attended_len = at::zeros({T}, opt);
    return f;
}

// The kept page set at a threshold: indices whose score >= tau.
std::vector<int64_t> kept_at(const at::Tensor& scores, double tau) {
    auto a = scores.to(at::kCPU, at::kDouble).contiguous();
    auto acc = a.accessor<double, 1>();
    std::vector<int64_t> out;
    for (int64_t i = 0; i < a.numel(); ++i) {
        if (acc[i] >= tau) {
            out.push_back(i);
        }
    }
    return out;
}

// Recall consumes the ORDER, not the scores, so the batched path must rank any pair
// the reference separates by more than the numeric tolerance the same way. A pair
// closer than that may legitimately swap under bf16 rounding.
void check_ranking_agrees(const at::Tensor& ref, const at::Tensor& got, double tol) {
    auto ra = ref.accessor<double, 1>();
    auto ga = got.accessor<double, 1>();
    for (int64_t i = 0; i < ref.numel(); ++i) {
        for (int64_t j = 0; j < ref.numel(); ++j) {
            if (ra[i] > ra[j] + tol) {
                CHECK(ga[i] > ga[j]);
            }
        }
    }
}

// Assert the bf16 batched rerank matches the fp32 per-page reference. The batched score
// matmuls run in bf16 on tensor cores (a deliberate perf/quality tradeoff), so the
// scores are approximate, not bit-exact; tol is a few times the observed bf16 gap, not
// a bound loose enough to admit a reordering. The RANKING must match exactly, since
// that is what selection consumes.
void check_batched_matches(const AttnScoreRetriever& sc, const QkFixture& f, double tol) {
    at::Tensor ref = qk_reference(sc, f.query, f.pages, f.lse, f.log_attended_len);
    at::Tensor bat = sc.rerank_score_batched(
                           f.query.vecs,
                           f.cand_k,
                           f.lse,
                           f.log_attended_len,
                           zero_positions(f.log_attended_len.numel()),
                           UNROTATED,
                           UNROTATED
    )
                         .to(at::kCPU, at::kDouble)
                         .contiguous();
    REQUIRE(ref.numel() == bat.numel());
    auto ra = ref.accessor<double, 1>();
    auto ba = bat.accessor<double, 1>();
    // Guard against a vacuous comparison: a fixture well above parity saturates every
    // score at ~1, and the two paths then agree for free.
    double min_ref_score = ra[0];
    for (int64_t i = 0; i < ref.numel(); ++i) {
        min_ref_score = std::min(min_ref_score, ra[i]);
    }
    REQUIRE(min_ref_score < 0.95);
    double max_abs = 0.0;
    for (int64_t i = 0; i < ref.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(ra[i] - ba[i]));
    }
    CHECK(max_abs < tol);
    check_ranking_agrees(ref, bat, tol);
}

at::Tensor positions_at(int64_t n_rows, int64_t pos) {
    return at::full({n_rows}, pos, at::TensorOptions().dtype(at::kLong));
}

}  // namespace

TEST_CASE("qk rerank_score_batched matches the per-page reference", "[recall]") {
    at::manual_seed(0);
    const at::Device cpu = at::kCPU;

    SECTION("GQA group") {
        QkFixture f = make_qk_fixture(
            /*n_layers=*/4,
            /*n_docs=*/9,
            /*T=*/5,
            /*ps=*/16,
            /*H_q=*/8,
            /*H_kv=*/2,
            /*d=*/8,
            cpu,
            /*ref_level=*/6.0
        );
        check_batched_matches(AttnScoreRetriever{}, f, 2e-3);
    }
    SECTION("a high reference, well below saturation") {
        QkFixture f = make_qk_fixture(3, 6, 4, 16, 8, 2, 8, cpu, 9.0);
        check_batched_matches(AttnScoreRetriever{}, f, 2e-3);
    }
    SECTION("same-head count (no GQA group)") {
        QkFixture f = make_qk_fixture(2, 5, 3, 16, 4, 4, 8, cpu, 6.0);
        check_batched_matches(AttnScoreRetriever{}, f, 2e-3);
    }

    // On a CUDA device the bf16 tensor-core batched matmul must stay CLOSE to the fp32
    // CPU per-page reference (approximate, not bit-exact; recall selects by ranking).
    if (at::hasCUDA()) {
        at::manual_seed(1);
        QkFixture ref = make_qk_fixture(4, 9, 5, 16, 8, 2, 8, at::kCPU, 6.0);
        // Reuse the same numeric values on device by copying the CPU fixture.
        QkFixture dev = ref;
        for (auto& t : dev.query.vecs) {
            t = t.to(at::kCUDA);
        }
        for (auto& t : dev.cand_k) {
            t = t.to(at::kCUDA);
        }
        for (auto& t : dev.lse) {
            t = t.to(at::kCUDA);
        }
        dev.log_attended_len = dev.log_attended_len.to(at::kCUDA);
        AttnScoreRetriever sc;
        at::Tensor cpu_ref = qk_reference(sc, ref.query, ref.pages, ref.lse, ref.log_attended_len);
        at::Tensor gpu = sc.rerank_score_batched(
                               dev.query.vecs,
                               dev.cand_k,
                               dev.lse,
                               dev.log_attended_len,
                               zero_positions(dev.log_attended_len.numel()),
                               UNROTATED,
                               UNROTATED
        )
                             .to(at::kCPU, at::kDouble)
                             .contiguous();
        auto ca = cpu_ref.accessor<double, 1>();
        auto ga = gpu.accessor<double, 1>();
        double max_abs = 0.0;
        for (int64_t i = 0; i < cpu_ref.numel(); ++i) {
            max_abs = std::max(max_abs, std::abs(ca[i] - ga[i]));
        }
        CHECK(max_abs < 5e-3);  // bf16 tensor-core rounding vs fp32 CPU reference
        check_ranking_agrees(cpu_ref, gpu, 5e-3);
    }
}

// The rerank rotates each query row to the position that row holds in the buffer and
// each candidate key from where it is stored to where it would sit once promoted, so a
// logit depends only on the GAP between the two. A query at `gap` against keys scored
// where they are stored must therefore score exactly what a query at `gap + offset`
// scores against keys rotated forward by `offset`, and a different gap must score
// differently.
TEST_CASE("rerank scores on the query-to-candidate gap", "[recall]") {
    at::manual_seed(3);
    AttnScoreRetriever scorer;
    QkFixture f = make_qk_fixture(
        /*n_layers=*/2,
        /*n_docs=*/6,
        /*T=*/4,
        /*ps=*/16,
        /*H_q=*/8,
        /*H_kv=*/2,
        /*d=*/8,
        at::kCPU,
        /*ref_level=*/6.0
    );
    const int64_t rows = f.log_attended_len.numel();
    const int64_t gap = 37, offset = 128;

    auto score_with = [&](const at::Tensor& query_positions, int64_t stored, int64_t scored) {
        return scorer
            .rerank_score_batched(f.query.vecs, f.cand_k, f.lse, f.log_attended_len, query_positions, stored, scored)
            .to(at::kCPU, at::kDouble)
            .contiguous();
    };
    auto max_abs_diff = [](const at::Tensor& left, const at::Tensor& right) {
        REQUIRE(left.numel() == right.numel());
        auto left_a = left.accessor<double, 1>();
        auto right_a = right.accessor<double, 1>();
        double worst = 0.0;
        for (int64_t i = 0; i < left.numel(); ++i) {
            worst = std::max(worst, std::abs(left_a[i] - right_a[i]));
        }
        return worst;
    };

    at::Tensor plain = score_with(positions_at(rows, gap), UNROTATED, UNROTATED);
    at::Tensor shifted = score_with(positions_at(rows, gap + offset), UNROTATED, offset);
    at::Tensor other_gap = score_with(positions_at(rows, gap + offset), UNROTATED, UNROTATED);
    // The matmul runs in bf16, so equal gaps agree to rounding, not to the bit; a
    // different gap must move the score far past that rounding.
    const double same_gap_diff = max_abs_diff(plain, shifted);
    const double other_gap_diff = max_abs_diff(plain, other_gap);
    CHECK(same_gap_diff < 5e-3);
    CHECK(other_gap_diff > 10 * same_gap_diff);
}

// One tau must mean the same thing under either position layout, so both must score a
// candidate at the same query-to-candidate gap, n_working. The candidate is scored at
// the closest slot it could occupy once promoted:
//   contiguous: query at active_len - 1, candidate at working_lo - 1
//   compacted:  query at short_offset + n_working, candidate at short_offset
// and working_lo = max(n_sink, active_len - n_working), so both gaps are n_working.
TEST_CASE("both position layouts score a candidate at the n_working gap", "[recall]") {
    at::manual_seed(5);
    AttnScoreRetriever scorer;
    QkFixture f = make_qk_fixture(
        /*n_layers=*/2,
        /*n_docs=*/6,
        /*T=*/1,
        /*ps=*/16,
        /*H_q=*/8,
        /*H_kv=*/2,
        /*d=*/8,
        at::kCPU,
        /*ref_level=*/6.0
    );
    const int64_t n_sink = 100, n_working = 512, active_len = 33000;
    const int64_t short_offset = n_sink;
    const int64_t working_lo = pulsar::ActiveBuffer::working_start(active_len, n_sink, n_working);
    REQUIRE(working_lo == active_len - n_working);

    auto score_with = [&](int64_t query_position, int64_t scored) {
        return scorer
            .rerank_score_batched(
                f.query.vecs,
                f.cand_k,
                f.lse,
                f.log_attended_len,
                positions_at(1, query_position),
                short_offset,
                scored
            )
            .to(at::kCPU, at::kDouble)
            .contiguous();
    };

    // Contiguous: the last slot queries, the candidate sits just under the working run.
    at::Tensor spread = score_with(active_len - 1, working_lo - 1);
    // Compacted: the working run ropes from short_offset + 1, so its last slot is
    // short_offset + n_working, and the candidate stays where demotion put it.
    at::Tensor collapsed = score_with(short_offset + n_working, short_offset);
    REQUIRE(spread.numel() == collapsed.numel());
    auto spread_a = spread.accessor<double, 1>();
    auto collapsed_a = collapsed.accessor<double, 1>();
    double worst_relative = 0.0;
    for (int64_t i = 0; i < spread.numel(); ++i) {
        worst_relative = std::max(worst_relative, std::abs(spread_a[i] - collapsed_a[i]) / std::abs(collapsed_a[i]));
    }
    CHECK(worst_relative < 1e-2);  // bf16 matmuls, so equal gaps agree to rounding
}
