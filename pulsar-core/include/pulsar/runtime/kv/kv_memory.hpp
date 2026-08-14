#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"
#include "pulsar/runtime/kv/file_bucket.hpp"
#include "pulsar/runtime/kv/page_view.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"
#include "pulsar/runtime/kv/paged_pool.hpp"
#include "pulsar/runtime/kv/recall.hpp"

#include <concepts>
#include <cstdint>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// KVMemory: the whole conversation's KV, held across an ActiveBuffer (attended KV
// the attention kernels read) and an ordered chain of spill buckets behind it, plus
// a RecallBuffer that stages one cycle's candidates for the reranker, plus the
// address-cursor read layer (the active/history PageViews). A token's ADDRESS is its
// index in the whole conversation; page index = address / page_size is a page's
// permanent identity.
//
// This is not a cache: the ActiveBuffer and the buckets are EXCLUSIVE (each page in
// exactly one of them), the demoted KV is the only copy, and dropping a page loses
// information. Per sequence they are complete:
//
//     ActiveBuffer united with s1 united with s2 == pages [0, next_page_index)
//
// The RecallBuffer sits OUTSIDE that invariant: redundant, droppable, and kept across
// cycles. A promotion removes a page from its BUCKET and also drops the buffer's copy.
//
// Each component is self-contained: it exposes a std::map-shaped cursor over its pages
// keyed by page index (the PageIndexed concept) plus its own accept surface. KVMemory
// owns the routing.

namespace pulsar {

// -- Active view: a per-sequence cursor over ActiveBuffer's page_id/tok -----------
// A non-owning view that binds one sequence of the multi-tenant ActiveBuffer so it
// satisfies PageIndexed. page_id[] is ascending in page position, so lower_bound is
// a binary search.
struct ActiveCursor {
    const ActiveBuffer* buf;
    int64_t seq;
    int64_t pp;  // page position

    PageEntry operator*() const {
        return PageEntry(
            this->buf->page_id(this->seq, this->pp),
            PageRef{&this->buf->page_token_ids(this->seq, this->pp)}
        );
    }
    ActiveCursor& operator++() {
        ++this->pp;
        return *this;
    }
    bool operator==(const ActiveCursor& o) const {
        return this->pp == o.pp && this->seq == o.seq && this->buf == o.buf;
    }
};

struct ActiveView {
    const ActiveBuffer* buf;
    int64_t seq;

    ActiveCursor lower_bound(int64_t idx) const {
        int64_t lo = 0, hi = this->buf->num_active_pages(this->seq);
        while (lo < hi) {
            const int64_t mid = lo + (hi - lo) / 2;
            if (this->buf->page_id(this->seq, mid) < idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return ActiveCursor{this->buf, this->seq, lo};
    }
    ActiveCursor end() const {
        return ActiveCursor{this->buf, this->seq, this->buf->num_active_pages(this->seq)};
    }
};
static_assert(PageIndexed<ActiveView>);

// -- The SpillBucket concept ------------------------------------------------------
// Medium-agnostic: nothing in it names RAM, disk, or a position. S1/S2 are ordered
// positions in KVMemory and either could hold either implementation.
//
// Two invariants the concept cannot express: buckets are ordered cheapest-first,
// and a bucket gather may be orders of magnitude more expensive than a pool
// gather.
template <typename B>
concept SpillBucket = requires(
    B& b,
    const B& cb,
    int64_t seq,
    int64_t step,
    int64_t budget_tokens,
    double decay,
    const ActiveBuffer& active,
    const DemotedKV& demoted,
    const Reservation& res,
    const PageKV& page,
    const std::vector<int64_t>& page_ids,
    const std::vector<int64_t>& layers
) {
    { cb.enabled() } -> std::same_as<bool>;
    { cb.view(seq) } -> PageIndexed;
    b.begin(seq);
    b.end(seq);
    { b.reserve(seq, demoted, step) } -> std::same_as<Reservation>;
    b.fill(seq, active, res);
    b.accept(seq, page);
    { cb.gather_k(seq, page_ids, layers) } -> std::same_as<GatheredK>;
    { b.take_pages(seq, page_ids) } -> std::same_as<std::vector<PageKV>>;
    { b.spill_to_budget(seq, budget_tokens, step, decay) } -> std::same_as<std::vector<PageKV>>;
    b.release_pending(seq);
    { cb.bucket_tokens(seq) } -> std::same_as<int64_t>;
    { cb.size() } -> std::same_as<int64_t>;
};
static_assert(SpillBucket<PagedBucket>);
static_assert(SpillBucket<FileBucket>);

// A read-only view of one sequence's pages in address order: (page index, page
// handle). Owns any snapshots its PageRefs point into (the bucket cursors for a
// merged history view, empty for the active-only view), so it is safe to return by
// value.
// Backed by a vector, so it models a random_access_range of PageEntry:
// `view | std::views::keys` yields page indices, `view | std::views::values` yields
// page handles (PageRef::token_ids(), a reference into the backing storage).
struct PageView {
    BucketView s1;  // owned s1 snapshot (empty for the active view)
    BucketView s2;  // owned s2 snapshot (empty for the active view)
    // Ascending by page index; PageRef -> s1 / s2 / the active buffer.
    std::vector<PageEntry> pages;

    std::vector<PageEntry>::const_iterator begin() const {
        return this->pages.begin();
    }
    std::vector<PageEntry>::const_iterator end() const {
        return this->pages.end();
    }
};

// One segment's BM25 document: the token ids at the segment's addresses, plus the
// tensor the retriever reads. A document is IMMUTABLE once its segment fills, so
// only the final partial segment ever rebuilds; `terms` is the cache and `dirty`
// marks it stale.
struct SegmentDoc {
    std::vector<int64_t> ids;  // by offset within the segment; -1 = never recorded
    at::Tensor terms;  // int64 [n_terms], the -1 placeholders dropped
    bool dirty = true;

    // The document tensor, rebuilding it only after an append.
    const at::Tensor& document();
};

// Recall SELECTION step: given per-page scores (fp64 [N], the unbounded would-be share
// r; -inf = a page selection may not take) and each page's token count, return the ROW
// indices into `scores` to promote, best first. This is where the keep-bar min_score
// (tau) is applied, and it is applied exactly once -- the rerank returns its scores
// unfiltered.
//
// tau is in [0, 1] but the scores are not: tau is mapped to r_bar = tau / (1 - tau) and
// THAT is compared against the score. Each candidate is promoted by its OWN independent
// Bernoulli draw, walking the candidates in descending score (ties by ascending row
// index):
//     keep_probability = sigmoid((log r - log r_bar) / temperature)
//   - temperature == 0 is the hard step at the bar: keep iff score >= r_bar. Special-
//     cased, so never a division by zero.
//   - tau <= 0 passes every candidate and tau >= 1 passes none; both short-circuit
//     before the draw, as does a score <= 0, so no NaN reaches the comparison.
//   - rng is touched only when keep_probability is strictly between 0 and 1.
//   - The unit of promotion is the winner plus every candidate page within block_radius
//     page ids of it, so an unscored page (-inf) can still be promoted as a neighbour
//     even though it can never start a promotion of its own. Neighbours already taken
//     are free; a page pulled in as a neighbour never draws.
//   - The stop condition is TOKEN capacity, not a page count: a neighbourhood whose
//     untaken tokens would push the promoted total past capacity_tokens is skipped
//     WHOLE and the walk continues, since half a neighbourhood is a different request.
//     Pages hold page_size tokens except a short tail page.
//   - pid must be ascending, which the page_id-sorted candidate list guarantees.
// selected_scores, when non-null, is cleared and filled PARALLEL to the returned rows:
// the score that won each promotion, which for an expanded neighbour is its WINNER's
// score, not its own (a neighbour need not have one).
std::vector<int64_t> select_recall_pages(
    const at::Tensor& scores,
    const std::vector<int64_t>& page_tokens,
    const std::vector<int64_t>& pid,
    int64_t block_radius,
    int64_t capacity_tokens,
    double bar,
    double temperature,
    std::mt19937_64& rng,
    std::vector<double>* selected_scores = nullptr
);

// One recall cycle's accounting, for a caller folding it into a run total.
// staged_hits/staged_misses split the cycle's prefiltered candidates by RecallBuffer
// residency and size the cache. promoted_s1/promoted_s2 split the cycle's promotions
// by the bucket they came out of.
struct RecallCounts {
    int64_t staged_hits = 0;
    int64_t staged_misses = 0;
    int64_t promoted_s1 = 0;
    int64_t promoted_s2 = 0;
};

// The KV memory over an ActiveBuffer, a RecallBuffer and an ordered bucket chain.
// Holds references to the engine-owned components. The retriever, the
// residency-independent segment documents and the per-cycle bookkeeping are owned here,
// not in any component. It is MULTI-TENANT: config carries only what every session
// shares, and the per-session knobs arrive per call (RecallParams, note_tokens'
// segment_size).
struct KVMemory {
    // s1 and s2 are the bucket chain, cheapest first.
    // skip_fresh filters pages evicted in the current cycle out of that cycle's
    // candidate set (see recall).
    KVMemory(
        ActiveBuffer& active,
        RecallBuffer& recall_buffer,
        PagedBucket& s1,
        FileBucket& s2,
        RecallConfig config = {},
        bool skip_fresh = true
    );

    // The sequence's attended pages in address order as a PageView (page index ->
    // token ids).
    PageView active(int64_t seq) const {
        PageView pv;
        ActiveView va{&this->active_buf, seq};
        for (auto c = va.lower_bound(0), e = va.end(); !(c == e); ++c) {
            pv.pages.push_back(*c);
        }
        return pv;
    }

    // The whole conversation's pages in address order: the ordered merge of the
    // active buffer and the buckets by page index. They are disjoint and complete,
    // so the merge yields the contiguous pages [0, next). All token ids in RAM; disk
    // is never read. The returned PageView owns the bucket snapshots.
    PageView history(int64_t seq) const {
        PageView pv;
        pv.s1 = this->s1.view(seq);
        pv.s2 = this->s2.view(seq);
        ActiveView va{&this->active_buf, seq};
        auto ca = va.lower_bound(0);
        const auto ea = va.end();
        auto c1 = pv.s1.lower_bound(0);
        const auto e1 = pv.s1.end();
        auto c2 = pv.s2.lower_bound(0);
        const auto e2 = pv.s2.end();
        auto live_a = [&] { return !(ca == ea); };
        auto live_1 = [&] { return !(c1 == e1); };
        auto live_2 = [&] { return !(c2 == e2); };
        // Merge-walk: at each step take the smallest-index page among the live ones.
        while (live_a() || live_1() || live_2()) {
            const bool xa = live_a(), x1 = live_1(), x2 = live_2();
            const int64_t ka = xa ? (*ca).first : 0;
            const int64_t k1 = x1 ? (*c1).first : 0;
            const int64_t k2 = x2 ? (*c2).first : 0;
            if (xa && (!x1 || ka < k1) && (!x2 || ka < k2)) {
                pv.pages.push_back(*ca);
                ++ca;
            } else if (x1 && (!x2 || k1 < k2)) {
                pv.pages.push_back(*c1);
                ++c1;
            } else {
                pv.pages.push_back(*c2);
                ++c2;
            }
        }
        return pv;
    }

    // Take an evict()'s victims straight into s1: reserve (host-only) then fill (the
    // D2H). Records the pages as FRESH for this cycle, so recall_skip_fresh can keep
    // them out of the cycle's candidate set. The fill completes before this returns,
    // so the caller may reuse the ActiveBuffer victim slots it read from.
    void accept_evicted(int64_t seq, const DemotedKV& demoted, int64_t step);

    // End of the evict->recall cycle, after promote: return the slots both buckets
    // marked dead this cycle, run the s1 -> s2 spill to ram_bucket_size, and clear
    // the fresh set. Running the spill here gives the cycle exactly one writer window
    // with no live reader in it. No-op spill when ram_bucket_size <= 0 or s2 is
    // disabled.
    void end_cycle(int64_t seq, int64_t ram_bucket_size, int64_t step, double decay);

    // Drop every per-sequence state KVMemory owns (segment documents, fresh set, the
    // RecallBuffer's cached copies). The buckets are ended by their owner.
    void end_seq(int64_t seq);

    // Record the token ids of addresses [addr[i], ...) as they are born, into the
    // per-seq segment documents. A document is immutable once its segment fills;
    // only the final partial segment grows. Residency-independent: a document
    // describes what was SAID, not what is currently available.
    // segment_size (TOKENS, a multiple of page_size) is the session's, and it KEYS the
    // documents (segment id = address / segment_size). It is therefore fixed for a
    // sequence's life: a later call naming a different one is a hard error, since the
    // ids already recorded would be filed under a grouping nothing else uses.
    void
    note_tokens(int64_t seq, const std::vector<int64_t>& addr, const std::vector<int64_t>& toks, int64_t segment_size);

    // Token ids at addresses [start, end), read from the segment documents, so the
    // cost is the range and not the conversation. Every address in the range must
    // have been recorded by note_tokens.
    std::vector<int64_t> token_ids_in(int64_t seq, int64_t start, int64_t end) const;

    // Score candidates across the bucket chain (RAM token_ids), return the selected
    // pages as Payloads (best first), and promote (remove) their pages from whichever
    // bucket holds them, reading full KV out of it. The candidate/return unit is a
    // single PAGE (one Payload per page); a segment is only the BM25 prefilter's tf/idf
    // document unit.
    // capacity_tokens is the promotion budget in TOKENS (not pages): selection stops
    // once the next candidate would not fit. capacity_tokens <= 0 returns nothing.
    // query_vecs are the query tokens' per-token Q projections (one
    // [q_len, n_heads, head_dim] per configured layer). lse is the captured softmax
    // denominator of those same query rows and log_attended_len their key counts (see
    // AttnScoreRetriever); both are required, as they are the score's reference.
    // query_positions are those rows' buffer positions, read AFTER the cycle's eviction
    // so they are the positions the rerank's keys will be scored against.
    //
    // params are the calling SESSION's recall knobs (see RecallParams); its
    // segment_size must be the one the sequence's documents were recorded under.
    //
    // Only kept-segment pages are scored, and selection runs on those RAW scores. When a
    // page wins its draw it is promoted together with every candidate page within
    // params.block_radius page ids of it, whatever bucket holds it and whether or not
    // its segment survived the prefilter: an expanded neighbour is recalled because it
    // is context, not because it scored. A radius of 0 promotes the winner alone.
    // raw_scores, when non-null, is filled with the scored pages' rerank scores as
    // scored, before selection; empty when no rerank runs.
    // counts, when non-null, is ADDED to with this cycle's staging and promotion
    // split (see RecallCounts).
    std::vector<Payload> recall(
        int64_t seq,
        const at::Tensor& query_token_ids,
        int64_t capacity_tokens,
        const std::vector<at::Tensor>& query_vecs,
        const std::vector<at::Tensor>& lse,
        const at::Tensor& log_attended_len,
        const at::Tensor& query_positions,
        const RecallParams& params,
        std::vector<double>* raw_scores = nullptr,
        RecallCounts* counts = nullptr
    );

    // A sequence's demoted tokens across the bucket chain (the RecallBuffer holds
    // copies and is not counted).
    int64_t demoted_tokens(int64_t seq) const {
        return this->s1.bucket_tokens(seq) + this->s2.bucket_tokens(seq);
    }

  private:
    // One candidate page: which bucket holds it, its page index, and a non-owning
    // handle to its RAM token ids (valid until a take_pages on that bucket).
    struct Cand {
        int64_t bucket;  // 0 = s1, 1 = s2
        int64_t page_id;
        const std::vector<int64_t>* token_ids;
    };

    // Split the given rows of the scored set by the bucket that owns them:
    // by_rows[b] are the row indices into the scored set, by_pids[b] the matching
    // page ids, both ascending. `rows` selects which rows to partition.
    static void partition_by_bucket(
        const std::vector<Cand>& pages,
        const std::vector<int64_t>& scored_pages,
        const std::vector<int64_t>& rows,
        std::vector<std::vector<int64_t>>& by_rows,
        std::vector<std::vector<int64_t>>& by_pids
    );

    // Make the scored pages' relevance-layer K resident in the RecallBuffer: bind
    // the cycle's scored set, ask each bucket for the MISSING rows it owns in one
    // batched gather, and scatter them into their cache slots. Cached rows cost no
    // transfer. Returns the per-configured-layer candidate tensors [n_scored,
    // page_size, n_kv_heads, head_dim] on the buffer's device, in scored order.
    // counts, when non-null, is added to with the hit/miss split.
    std::vector<at::Tensor> stage_candidates(
        int64_t seq,
        const std::vector<Cand>& pages,
        const std::vector<int64_t>& scored_pages,
        RecallCounts* counts
    );

    // Rerank on the GPU: score every candidate page in one batched matmul
    // (AttnScoreRetriever::rerank_score_batched) against the caller-staged cand_k (per
    // configured layer [n_scored, page_size, n_kv_heads, head_dim] on the pool device).
    // Keeps the query on device and returns every score in scored order, unfiltered.
    // Returns CPU fp64 [n_scored]. query_vecs are the on-device query Q per configured
    // layer; query_positions are the buffer positions of the query rows;
    // stored/scored_key_position are the calling session's (see RecallParams).
    at::Tensor rerank_gpu(
        const std::vector<at::Tensor>& query_vecs,
        const std::vector<at::Tensor>& cand_k,
        const std::vector<at::Tensor>& lse,
        const at::Tensor& log_attended_len,
        const at::Tensor& query_positions,
        int64_t stored_key_position,
        int64_t scored_key_position
    );

    // One sequence's BM25 corpus: the grouping it was recorded under and the documents
    // keyed by segment id (address / segment_size). The size is stored WITH the
    // documents because it is what the keys mean.
    struct SegmentIndex {
        int64_t segment_size = 0;
        std::map<int64_t, SegmentDoc> docs;
    };

    ActiveBuffer& active_buf;
    RecallBuffer& recall_buffer;
    PagedBucket& s1;
    FileBucket& s2;
    RecallConfig config;
    // Pages evicted in the current cycle are filtered out of that cycle's candidate
    // set, so the rerank never reads a page whose fill may still be in flight.
    bool skip_fresh;
    CascadeRetriever retriever;
    // Recall-selection RNG (the per-page Bernoulli draws). Seeded once from config.seed;
    // persists across recall() calls so a run is reproducible for a given seed.
    std::mt19937_64 rng;
    // Per-seq page indices evicted in the current cycle; cleared by end_cycle.
    std::unordered_map<int64_t, std::unordered_set<int64_t>> fresh;
    // Per-seq segment documents: segment id -> the token ids at that segment's
    // addresses, appended as tokens are born. Residency-independent, so they
    // duplicate the ids the pages hold in tok[].
    std::unordered_map<int64_t, SegmentIndex> segments;
};

}  // namespace pulsar
