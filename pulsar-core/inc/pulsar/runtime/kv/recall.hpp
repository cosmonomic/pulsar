#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"  // DemotedKV
#include "pulsar/runtime/kv/paged_pool.hpp"  // PageKV, PagedPool

#include <ATen/core/Tensor.h>

#include <ATen/core/List.h>
#include <c10/core/Device.h>

#include <concepts>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Recall: the RecallBuffer (a persistent device cache of candidate pages'
// relevance-layer K) and the retriever scorers. Recall is a two-stage cascade: a BM25
// prefilter groups pages into fixed-size SEGMENTS (recall_segment_size tokens, a
// multiple of page_size) as tf/idf documents and keeps the top-scoring segments; a qk
// attention reranker then scores and returns individual PAGES (segments are only the
// prefilter's document unit).
// KVMemory::recall scores candidates by the cascade against a query and returns the
// selected pages as Payloads to recompact back into the ActiveBuffer.
//
// A token's identity is its conversation address (int64, monotonic): the prompt
// occupies addr [0, prompt_len); the i-th generated token is addr prompt_len + i.
// A page's index is address / page_size; its segment id is its base address /
// segment_size. Plain C++, engine-internal; not TorchBind-exposed. Unit tests in
// pulsar-core/tests/runtime/.

namespace pulsar {

// One recalled page's full KV, read out of the bucket that held it.
// addr/token_ids/pos are CPU, ascending in addr; pos is the position the page's K is
// actually roped at, the session's short_offset, which recompact re-ropes from.
// k/v are the holding bucket's page-major carriers, contiguous and page-locked when
// it holds them in RAM, so the promotion uploads each page in one transfer.
struct Payload {
    int64_t base_addr = 0;  // conversation address of row 0; the page spans [base, base+B)
    at::Tensor token_ids;  // int64 [B]
    at::Tensor k;  // [B, n_layers, n_kv_heads, head_dim] (pool layout)
    at::Tensor v;  // [B, n_layers, n_kv_heads, head_dim] (pool layout)
    // The rerank density that won this page its promotion: its own score when it won its
    // own draw, its winner's score when a winner's block_radius expansion pulled it in.
    double score = 0.0;

    int64_t num_tokens() const {
        return this->token_ids.numel();
    }
};

// A query or candidate segment as the scorers see it: token ids (for lexical
// scorers) and per-token vectors (for the QK-attention scorer). vecs holds one
// tensor per configured layer, in RecallConfig::layers order: for a query the query
// tokens' Q projections [q_len, n_heads, head_dim]; for a segment its stored K
// [seg_len, n_kv_heads, head_dim]. Empty when no vectors (BM25-only).
struct RetrievalDoc {
    at::Tensor token_ids;  // int64 [len]
    std::vector<at::Tensor> vecs;  // per configured layer; per-token, empty for BM25
};

// BM25 of the query token ids against each candidate segment's token ids, fp64
// [num_segments] and higher is better (the prefilter argsorts descending). IDF over the
// candidate segments. Double-precision, fixed op order. Ignores the docs' vecs.
struct Bm25Retriever {
    at::Tensor score(const RetrievalDoc& query, const std::vector<RetrievalDoc>& segments) const;
};

// The cascade secondary. Scores each candidate page by an UNBOUNDED mass-space
// attention score. Query Q arrives pre-RoPE and is rotated to the query row's own
// buffer position; candidate K is stored roped at the session's short_offset and is
// rotated from there to the position the page would occupy once promoted. GQA maps
// query head h to kv head h / (H_q/H_kv). Ignores token_ids; needs vecs.
struct AttnScoreRetriever {
    // The model's own q.k multiplier. Must match the forward pass: the score's
    // denominator is the kernel's LSE, computed with it.
    double scale = -1.0;  // < 0 => 1 / sqrt(head_dim)
    // The model's RoPE frequency base, so the query rotation matches the forward pass.
    // The POSITIONS rotated to are per call (see rerank_score_batched).
    double rope_theta = 10000.0;
    // The length in TOKENS the density is stated against. Must be the length the
    // EVICTION density is stated against (ActiveBuffer::set_mass_reference_length), or
    // the two sides of the keep-bar are in different units. <= 0 uses each query row's
    // own attended key count, the length the attention kernels applied themselves.
    int64_t mass_reference_length = 0;

    // Returns the candidate's would-be attention DENSITY, in the same units
    // ActiveBuffer::evict compares against its keep-bar: an approximate softmax weight
    // times the kernel's own length gain. UNBOUNDED, not a squashed score.
    // Per (query token, query head, configured layer, page slot), the candidate's logit
    // against the ACTIVE buffer's denominator gives an approximate softmax weight, which
    // then takes the same length gain the kernel applies to a resident key:
    //   density = exp(scale * q.k - lse) * length
    // for length the query row's own attended_len, or mass_reference_length when that is
    // > 0. Reduced by MEAN over query tokens, then MEAN over query heads, then MAX over
    // the page's slots, then MAX over layers -- the reduction order ActiveBuffer::evict
    // uses.
    //
    // Inputs are pre-stacked and may live on any device. The score matmuls run in bf16
    // on tensor cores (fp32 accumulation), so the returned scores are approximate. All
    // candidate pages must share the same token count ps (recall scores whole page_size
    // pages), so they stack.
    //   query_vecs:   per configured layer [T, H_q, head_dim]
    //   cand_k:       per configured layer [n_docs, ps, H_kv, head_dim] (page K)
    //   lse:          per configured layer [T, H_q], the softmax denominator the
    //                 attention itself used for each query row. Required: it is the
    //                 score's reference.
    //   log_attended_len: [T] fp32, log of the keys each query row attended. Unread
    //                 when mass_reference_length is > 0, which replaces it.
    //   query_positions: int64 [T], the buffer position of each query row. Row i is
    //                    rotated to its own angle.
    //   stored_key_position: the position cand_k is roped at, the session's short_offset.
    //   scored_key_position: the position to score the candidate at. The keys take one
    //                    delta rotation from stored to scored; equal positions rotate
    //                    nothing.
    // Returns CPU fp64 [n_docs].
    at::Tensor rerank_score_batched(
        const std::vector<at::Tensor>& query_vecs,
        const std::vector<at::Tensor>& cand_k,
        const std::vector<at::Tensor>& lse,
        const at::Tensor& log_attended_len,
        const at::Tensor& query_positions,
        int64_t stored_key_position,
        int64_t scored_key_position
    ) const;
};

// Two-stage cascade retriever. PREFILTER: BM25 over all N segments keeps the top
// prefilter_k (M); the other N-M get -inf. RERANK: the secondary scores each kept page.
// The rerank does NOT apply a keep-bar: it returns every kept page's score as scored. The
// bar (RecallConfig::min_score) is applied once, in the SELECTION step
// (select_recall_pages), on the raw scores.
struct CascadeRetriever {
    Bm25Retriever bm25;
    AttnScoreRetriever secondary;

    // Score BM25 over all segments (token_ids only) and return the top prefilter_k
    // segment indices in BM25-descending order. The caller gathers only those winners'
    // pages for the secondary and scatters their scores back to the full segment list
    // (non-winners -inf). prefilter_k is the width M, per call because it is a SESSION
    // knob; <= 0 takes every segment.
    //
    // Scores are RAW: segments are fixed-boundary documents and the prefilter ranks them
    // as they stand. Contiguity is not the prefilter's job -- selection expands a chosen
    // page over its page_id neighbourhood, across segment boundaries.
    // BM25 scores EVERY segment, so idf is the statistic over the whole corpus.
    // selectable (empty => all) restricts which segments may fill the top M: a
    // segment that can contribute no candidate is skipped rather than consuming a
    // slot.
    std::vector<int64_t> prefilter(
        const RetrievalDoc& query,
        const std::vector<RetrievalDoc>& segments,
        int64_t prefilter_k,
        const std::vector<char>& selectable = {}
    ) const;
};

// Retriever + vector configuration, all of it ENGINE-level: one model, one pool
// geometry and one RNG stream serve every session. layers are the attention layers the
// reranker scores. scale is the model's q.k multiplier (< 0 => 1 / sqrt(head_dim)).
// The per-session knobs are not here; they travel per call (see RecallParams).
struct RecallConfig {
    std::vector<int64_t> layers;
    double scale = -1.0;
    // The model's RoPE frequency base (see AttnScoreRetriever).
    double rope_theta = 10000.0;
    // The length in TOKENS the density is stated against, the same one the eviction
    // side uses (see AttnScoreRetriever). <= 0 uses each query row's own attended key
    // count.
    int64_t mass_reference_length = 0;
    // Seeds the recall-selection RNG once (>= 0).
    int64_t seed = 0;
};

// The recall knobs a SESSION owns, passed per call because two live sessions may hold
// different ones. Resolved by the caller (Engine::resolve_session): the bars are already
// density bars, not taus, and segment_size is already in tokens.
struct RecallParams {
    // The keep-bar, in density units. This is the SAME quantity ActiveBuffer::evict
    // compares against, derived from tau by evict_bar_from_tau: 0 recalls every scored
    // page, +inf recalls none.
    double bar = 0.0;
    // Selection temperature: 0 => the bar is a hard step (recall a page iff its score >=
    // the bar); > 0 => each page is promoted by an independent Bernoulli draw whose
    // probability is the bar's log-odds distance squashed at this temperature (see
    // select_recall_pages).
    double temperature = 0.0;
    // BM25 prefilter width M; <= 0 => no prefilter.
    int64_t prefilter_k = -1;
    // The absolute position in TOKENS a demoted page's K is roped at, resolved by the
    // caller. Recorded on every promoted Payload.
    int64_t short_offset = 0;
    // The buffer position the rerank scores a candidate at: the closest slot the page
    // could occupy once promoted. Equal to short_offset under the compacted layout.
    int64_t candidate_position = 0;
    // The BM25 prefilter's tf/idf document unit, in TOKENS: a multiple of page_size, and
    // the same value the session's documents were recorded under (see
    // KVMemory::note_tokens).
    int64_t segment_size = 0;
    // Neighbourhood radius in PAGES a winner's promotion expands over; 0 promotes the
    // winner alone.
    int64_t block_radius = 0;
};

// A persistent device cache of candidate pages' relevance-layer K, keyed by
// (sequence, page index). Relevance-layer K ONLY: that is all rerank_score_batched
// reads, and a promotion reads full KV from the bucket instead.
//
// It holds redundant COPIES, so it sits OUTSIDE the completeness invariant (ActiveBuffer
// united with the buckets) and is fully decoupled from bucket slot lifetime: it never
// interacts with a bucket's reserve/fill/release_pending or the end-of-cycle fence.
//
// A cached copy stays valid for the page's whole demoted lifetime, so an s1 -> s2 spill
// must NOT drop it. Leaving the buckets (a promotion) is the only event that
// invalidates one, via drop().
//
// The pool is preallocated to recall_buffer_size and never grows. Its capacity is a
// hard FLOOR at the worst-case scored set: a cycle's whole candidate set must be
// simultaneously resident, so a larger cycle is an error and never a clamp.
struct RecallBuffer {
    // n_rel_layers is the number of RELEVANCE layers (RecallConfig::layers), not the
    // model's layer count: entries are indexed by position in that list.
    //
    // cycle_pages is ONE sequence's staging budget (recall_buffer_size / page_size) and
    // bounds a single cycle's scored set. The pool is max_sessions of them, the same way
    // ActiveBuffer is max_sessions whole windows, so every live session's cycle fits at
    // once and no session's staging can be starved by another's. Entries persist across
    // cycles and the LRU is deliberately GLOBAL over that pool: a session's cycle is
    // guaranteed, its cache is best-effort, so an idle session's stale pages are
    // reclaimable by an active one.
    RecallBuffer(
        int64_t n_rel_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        at::ScalarType dtype,
        int64_t cycle_pages,
        int64_t max_sessions,
        at::Device device = at::kCPU
    );
    // String-dtype ctor ("float16"/"bfloat16"/"float32"), for the C++ unit tests.
    RecallBuffer(
        int64_t n_rel_layers,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t page_size,
        std::string dtype,
        int64_t cycle_pages,
        int64_t max_sessions,
        at::Device device = at::kCPU
    );
    ~RecallBuffer();
    // Movable (defined in recall.cpp where PagedPool is complete). Not copyable.
    RecallBuffer(RecallBuffer&&) noexcept;
    RecallBuffer& operator=(RecallBuffer&&) noexcept;

    int64_t capacity_pages() const;
    int64_t capacity_tokens() const;
    int64_t staged_pages() const;  // rows in the current cycle's scored set
    int64_t cached_pages() const;  // entries resident across cycles

    // Bind seq's scored set for this cycle: page_ids in SCORED order. Resident pages
    // are reused and become most-recently-used; the rest are given a slot, evicting
    // the least-recently-used entries once the pool is full. Returns the ROW
    // positions within page_ids that missed and must be filled from a bucket. Hard
    // error when the scored set exceeds the pool: the buffer never grows and a cycle
    // is never silently clamped.
    std::vector<int64_t> begin_cycle(int64_t seq, const std::vector<int64_t>& page_ids);
    // Copy src [n, page_size, n_kv_heads, head_dim] (any device/dtype) into the cache
    // slots of `rows` for relevance layer li. Rows must be MISS rows of the current
    // cycle; a hit is already resident.
    void fill(int64_t li, const std::vector<int64_t>& rows, const at::Tensor& src);
    // The cycle's candidate K for relevance layer li as
    // [staged_pages(), page_size, n_kv_heads, head_dim] in SCORED order, on the pool
    // device and dtype: the batched reranker's input. Scattered residency makes this
    // a device gather; every miss row must have been filled for this layer first.
    at::Tensor candidates(int64_t li) const;

    // Drop a page's cached copy, because it LEFT the buckets (a promotion). A spill
    // between buckets must not call this. Safe mid-cycle: the freed slot is not
    // reused before the next begin_cycle.
    void drop(int64_t seq, int64_t page_id);
    // Drop every entry a sequence holds.
    void end_seq(int64_t seq);

  private:
    // A cached page's identity. page_id alone is per-sequence, so entries from
    // different sequences never collide.
    struct Key {
        int64_t seq = 0;
        int64_t page_id = 0;
        bool operator==(const Key& o) const {
            return this->seq == o.seq && this->page_id == o.page_id;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return std::hash<int64_t>{}(k.seq) * 1000003u ^ std::hash<int64_t>{}(k.page_id);
        }
    };
    // A cached page: its pool slot and its node in the recency order.
    struct Entry {
        int64_t pool_page = 0;
        std::list<Key>::iterator recency;
    };

    // Free a slot back to the pool and forget the entry.
    void erase(std::unordered_map<Key, Entry, KeyHash>::iterator it);

    std::unique_ptr<PagedPool> pool;
    int64_t cycle_pages;  // one sequence's staging budget; bounds a cycle's scored set
    std::unordered_map<Key, Entry, KeyHash> entries;
    std::list<Key> recency;  // most-recently-used first; the back evicts next
    // This cycle's scored set: the pool slot behind each row, in scored order.
    std::vector<int64_t> cycle_slots;
    std::vector<char> cycle_miss;  // per row: staged this cycle rather than resident
    std::vector<int64_t> cycle_filled;  // per relevance layer: miss rows filled
    int64_t n_miss = 0;
    // Token-level slot indices of cycle_slots, on the pool device: candidates()
    // gathers by it. Built once per cycle and shared by every relevance layer.
    at::Tensor gather_index;
};

}  // namespace pulsar
