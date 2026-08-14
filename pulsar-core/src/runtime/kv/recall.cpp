#include "pulsar/runtime/kv/recall.hpp"
#include "pulsar/profiling.hpp"
#include "pulsar/rope.hpp"

#include "host_pages.hpp"
#include "../tensor_util.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <vector>

// The RecallBuffer's device staging pool + the retriever scorers. The buffer is a
// persistent (seq, page_id)-keyed LRU cache of prefiltered pages' relevance-layer K
// in one flat PagedPool of whole address-aligned pages; a cycle stages only its
// misses. BM25 arithmetic is in double with a fixed op order; ranking is by
// at::argsort.

namespace pulsar {

namespace {

constexpr double kBm25K1 = 1.5;
constexpr double kBm25B = 0.75;

}  // namespace

at::Tensor Bm25Retriever::score(const RetrievalDoc& query, const std::vector<RetrievalDoc>& segments) const {
    const int64_t n_docs = static_cast<int64_t>(segments.size());
    at::Tensor q_all_c = query.token_ids.to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> q_all(q_all_c.data_ptr<int64_t>(), q_all_c.data_ptr<int64_t>() + q_all_c.numel());
    std::unordered_set<int64_t> q_set(q_all.begin(), q_all.end());
    std::vector<std::vector<int64_t>> doc_ids(n_docs);
    std::vector<double> doc_lens(n_docs);
    double len_sum = 0.0;
    for (int64_t i = 0; i < n_docs; ++i) {
        at::Tensor doc_ids_c = segments[i].token_ids.to(at::kCPU, at::kLong).contiguous();
        doc_ids[i].assign(doc_ids_c.data_ptr<int64_t>(), doc_ids_c.data_ptr<int64_t>() + doc_ids_c.numel());
        doc_lens[i] = static_cast<double>(doc_ids[i].size());
        len_sum += doc_lens[i];
    }
    double avgdl = n_docs ? len_sum / static_cast<double>(n_docs) : 0.0;
    std::vector<double> scores(n_docs, 0.0);
    for (int64_t t : q_set) {
        std::vector<double> tf(n_docs, 0.0);
        int64_t df = 0;
        for (int64_t i = 0; i < n_docs; ++i) {
            int64_t c = 0;
            for (int64_t id : doc_ids[i]) {
                if (id == t) {
                    ++c;
                }
            }
            tf[i] = static_cast<double>(c);
            if (c > 0) {
                ++df;
            }
        }
        if (df == 0) {
            continue;
        }
        double idf = std::log(1.0 + (static_cast<double>(n_docs) - df + 0.5) / (df + 0.5));
        for (int64_t i = 0; i < n_docs; ++i) {
            double denom = tf[i] + kBm25K1 * (1.0 - kBm25B + kBm25B * doc_lens[i] / avgdl);
            scores[i] += idf * (tf[i] * (kBm25K1 + 1.0)) / denom;
        }
    }
    return at::tensor(scores, at::TensorOptions().dtype(at::kDouble));
}

// Mass-space attention score for candidate pages, one batched matmul per layer.
// page_best_logit = max over the page's keys of scale * q.k; lse is the softmax
// denominator the attention itself used for that query row, captured from the kernel,
// so it covers every key the row attended. With length that row's own attended key
// count, or mass_reference_length when one is configured,
// Per (layer, query head, query token, page slot), r = exp(scale * q.k - lse +
// log length) is that key's attention share as a multiple of a uniform share over that
// length, which is the density the eviction mass accumulates.
// The reductions therefore mirror ActiveBuffer::evict exactly, and in its
// order: MEAN over query tokens first, since a resident key's mass is a time-average
// and the slot max is applied to that average, not to each step's peak; then MEAN over
// query heads, MAX over the page's slots, MAX over layers. Reducing the slots first
// would score mean_t(max_slot) where eviction holds max_slot(mean_t), which is larger
// whenever different keys win at different query tokens; reducing them before the heads
// would score a page by a number belonging to no single key.
//
// The reduction RETURNS r itself, not a squashed score: r is the density the eviction
// bar is stated in, so selection compares it against the same bar in the same units.
// Squashing to r / (1 + r) is a reporting
// convenience only (parity -- exactly uniform -- at 0.5) and happens at the histogram,
// after selection: r / (1 + r) is monotone in r but not in the density's units, and the
// bar is stated in those units.
//
// Means over r are taken as logsumexp minus log n, the same quantity without overflowing
// exp.
//
// The matmuls run in bf16 on tensor cores with fp32 accumulation, so scores are
// approximate; the layer reduction is fp64 on the host.
//
// Shape annotations below use H_q = n_query_heads, H_kv = n_kv_heads, T = query
// tokens, ps = page_size, d = head_dim.
at::Tensor AttnScoreRetriever::rerank_score_batched(
    const std::vector<at::Tensor>& query_vecs,
    const std::vector<at::Tensor>& cand_k,
    const std::vector<at::Tensor>& lse,
    const at::Tensor& log_attended_len,
    const at::Tensor& query_positions,
    int64_t stored_key_position,
    int64_t scored_key_position
) const {
    PULSAR_PROF_SCOPE(RERANK_SCORE);
    const int64_t n_layers = static_cast<int64_t>(query_vecs.size());
    TORCH_CHECK(n_layers > 0, "AttnScoreRetriever::rerank_score_batched: query has no vectors");
    TORCH_CHECK(
        static_cast<int64_t>(cand_k.size()) == n_layers,
        "rerank_score_batched: cand_k has ",
        cand_k.size(),
        " layers, expected ",
        n_layers
    );
    const int64_t n_docs = cand_k[0].size(0);
    // The captured denominator is the score's reference; without it there is nothing
    // to score against, so a missing one is a caller error, not a fallback.
    TORCH_CHECK(
        static_cast<int64_t>(lse.size()) == n_layers && lse[0].defined(),
        "rerank_score_batched: need one captured lse per layer"
    );
    TORCH_CHECK(
        log_attended_len.defined() && log_attended_len.numel() == lse[0].size(0),
        "rerank_score_batched: log_attended_len must be one entry per query "
        "token"
    );
    TORCH_CHECK(
        query_positions.defined() && query_positions.numel() == lse[0].size(0),
        "rerank_score_batched: query_positions must be one position per query "
        "token"
    );

    // The relevance layers have no layer axis to reduce over: Q and candidate K arrive as
    // one tensor per layer, so the max over layers is folded across the loop. It
    // accumulates on the scoring device and crosses to the host once, after the last
    // layer; a D2H per layer would sync the whole rerank once per layer per cycle.
    at::Tensor best;
    {
        PULSAR_PROF_SCOPE(RERANK_BMM);
        for (int64_t layer_index = 0; layer_index < n_layers; ++layer_index) {
            at::Tensor q = query_vecs[layer_index].to(at::kFloat);  // [T, H_q, d]
            const int64_t n_query_heads = q.size(1);
            const int64_t d = q.size(2);
            const double scale = this->scale > 0.0 ? this->scale : 1.0 / std::sqrt(static_cast<double>(d));
            // Rotate each query row to ITS OWN buffer position.
            at::Tensor qr = rope_rotate(q, query_positions.to(q.device(), at::kDouble).view({-1, 1}), this->rope_theta);
            at::Tensor query_block = qr.transpose(0, 1).contiguous().to(at::kBFloat16);  // [H_q, T, d] bf16

            // [T, H_q] -> [H_q, T], matching page_best_logit's trailing axes.
            at::Tensor row_lse = lse[layer_index].to(q.device(), at::kFloat).transpose(0, 1);
            // The length gain the mass carries, per query row or one value for all.
            at::Tensor row_log_len = this->mass_reference_length > 0
                ? at::full({1, 1}, std::log(static_cast<double>(this->mass_reference_length)), row_lse.options())
                : log_attended_len.to(q.device(), at::kFloat).unsqueeze(0);  // [1, T]

            // Move the candidate from where it is stored to where it would sit once
            // promoted: one delta rotation over the staged keys. Equal positions leave
            // the staged tensor untouched.
            at::Tensor cand = cand_k[layer_index];
            if (scored_key_position != stored_key_position) {
                auto dopt = at::TensorOptions().dtype(at::kDouble).device(cand.device());
                cand = rope_rotate(
                    cand.to(at::kFloat),
                    at::scalar_tensor(static_cast<double>(scored_key_position - stored_key_position), dopt),
                    this->rope_theta
                );
            }
            at::Tensor cand_keys = cand.to(at::kBFloat16);  // [n_docs, ps, H_kv, d] bf16
            const int64_t group = n_query_heads / cand_keys.size(2);
            // [n_docs, ps, H_kv, d] -> [n_docs, H_q, ps, d]; head h -> kv head h / group.
            at::Tensor cand_key_block = cand_keys.transpose(1, 2).repeat_interleave(group,
                                                                                    1);  // [n_docs, H_q, ps, d]
            // page_logits[n_docs, H_q, T, ps] =
            //     query_block[1, H_q, T, d] @ cand_key_block^T[n_docs, H_q, d, ps].
            at::Tensor page_logits = at::matmul(query_block.unsqueeze(0), cand_key_block.transpose(2, 3))
                                         .to(at::kFloat) *
                scale;
            // exp(logit - lse) is the approximate softmax weight; the length gain is
            // the kernel's own, so the result is in the eviction density's units.
            // [n_docs, H_q, T, ps]; the per-row reference broadcasts over docs and slots.
            at::Tensor log_r = page_logits - (row_lse - row_log_len).unsqueeze(0).unsqueeze(-1);
            const int64_t n_tok = log_r.size(2), n_head = log_r.size(1);
            // mean over query tokens -> [n_docs, H_q, ps]; then the cross-head mean, then
            // the slot max. The head axis must collapse before the slot max so the page's
            // score belongs to a SINGLE key; max and mean do not commute.
            at::Tensor per_slot_head = at::logsumexp(log_r, /*dim=*/2) - std::log(static_cast<double>(n_tok));
            // Leave log space in fp64: a candidate key is not in the LSE's attended set,
            // so log_r has no upper bound, and fp32 exp overflows to +inf past ~88.
            at::Tensor per_slot = at::logsumexp(per_slot_head.to(at::kDouble), /*dim=*/1) -
                std::log(static_cast<double>(n_head));  // [n_docs, ps]
            at::Tensor per_layer = at::exp(std::get<0>(per_slot.max(/*dim=*/1)));
            best = best.defined() ? at::maximum(best, per_layer) : per_layer;
        }
    }

    return best.to(at::kCPU, at::kDouble).contiguous();
}

// PREFILTER: BM25 over all segments (token_ids only); return the top prefilter_k
// SELECTABLE segment indices in descending RAW-BM25 order. Segments are fixed-boundary
// documents, scored as they stand. prefilter_k <= 0 takes every segment. No vecs touched.
std::vector<int64_t> CascadeRetriever::prefilter(
    const RetrievalDoc& query,
    const std::vector<RetrievalDoc>& segments,
    int64_t prefilter_k,
    const std::vector<char>& selectable
) const {
    const int64_t n_docs = static_cast<int64_t>(segments.size());
    std::vector<int64_t> kept;
    if (n_docs == 0) {
        return kept;
    }
    at::Tensor bm = this->bm25.score(query, segments);
    const int64_t M = prefilter_k > 0 ? std::min<int64_t>(prefilter_k, n_docs) : n_docs;
    at::Tensor order = at::argsort(bm, /*dim=*/-1, /*descending=*/true);
    at::Tensor order_c = order.to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> order_h(order_c.data_ptr<int64_t>(), order_c.data_ptr<int64_t>() + order_c.numel());
    // A non-selectable segment scores but cannot contribute a candidate, so it is
    // skipped instead of consuming one of the M slots.
    kept.reserve(M);
    for (int64_t i : order_h) {
        if (static_cast<int64_t>(kept.size()) >= M) {
            break;
        }
        if (!selectable.empty() && !selectable[i]) {
            continue;
        }
        kept.push_back(i);
    }
    return kept;
}

RecallBuffer::RecallBuffer(
    int64_t n_rel_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    at::ScalarType dtype,
    int64_t cycle_pages,
    int64_t max_sessions,
    at::Device device
)
    : pool(
          std::make_unique<PagedPool>(
              n_rel_layers,
              n_kv_heads,
              head_dim,
              page_size,
              dtype,
              device,
              /*store_v=*/false,
              /*max_pages=*/cycle_pages * max_sessions,
              /*pinned=*/false,
              /*preallocate=*/true
          )
      ),
      cycle_pages(cycle_pages) {
    TORCH_CHECK(
        cycle_pages > 0,
        "RecallBuffer: cycle_pages must be > 0 (recall_buffer_size / "
        "page_size)"
    );
    TORCH_CHECK(max_sessions > 0, "RecallBuffer: max_sessions must be > 0");
}

RecallBuffer::RecallBuffer(
    int64_t n_rel_layers,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t page_size,
    std::string dtype,
    int64_t cycle_pages,
    int64_t max_sessions,
    at::Device device
)
    : RecallBuffer(
          n_rel_layers,
          n_kv_heads,
          head_dim,
          page_size,
          parse_dtype(dtype, "RecallBuffer"),
          cycle_pages,
          max_sessions,
          device
      ) {}

RecallBuffer::~RecallBuffer() = default;
RecallBuffer::RecallBuffer(RecallBuffer&&) noexcept = default;
RecallBuffer& RecallBuffer::operator=(RecallBuffer&&) noexcept = default;

int64_t RecallBuffer::capacity_pages() const {
    return this->pool->capacity;
}

int64_t RecallBuffer::capacity_tokens() const {
    return this->pool->capacity * this->pool->page_size;
}

int64_t RecallBuffer::staged_pages() const {
    return static_cast<int64_t>(this->cycle_slots.size());
}

int64_t RecallBuffer::cached_pages() const {
    return static_cast<int64_t>(this->entries.size());
}

void RecallBuffer::erase(std::unordered_map<Key, Entry, KeyHash>::iterator it) {
    this->pool->free_page(it->second.pool_page);
    this->recency.erase(it->second.recency);
    this->entries.erase(it);
}

std::vector<int64_t> RecallBuffer::begin_cycle(int64_t seq, const std::vector<int64_t>& page_ids) {
    const int64_t n = static_cast<int64_t>(page_ids.size());
    // Against the PER-SEQUENCE budget, not the whole pool: the pool holds max_sessions
    // of these, and a cycle that ate into another session's share would break the
    // guarantee that every live session's staging fits at once.
    TORCH_CHECK(
        n <= this->cycle_pages,
        "RecallBuffer: cycle scores ",
        n,
        " pages but one sequence's staging budget is ",
        this->cycle_pages,
        " (recall_buffer_size = ",
        this->cycle_pages * this->pool->page_size,
        " tokens); raise recall_buffer_size"
    );
    this->cycle_slots.assign(n, -1);
    this->cycle_miss.assign(n, 0);
    this->cycle_filled.assign(this->pool->n_layers, 0);
    std::vector<int64_t> miss;
    // Touch every hit first, so the whole cycle sits ahead of any eviction victim in
    // the recency order. n <= capacity then makes the back a non-member.
    for (int64_t r = 0; r < n; ++r) {
        auto it = this->entries.find(Key{seq, page_ids[r]});
        if (it == this->entries.end()) {
            miss.push_back(r);
            continue;
        }
        this->recency.splice(this->recency.begin(), this->recency, it->second.recency);
        this->cycle_slots[r] = it->second.pool_page;
    }
    for (int64_t r : miss) {
        // The pool is preallocated, so an empty free list means every slot is a live
        // entry: make room by dropping the least-recently-used one.
        if (this->pool->free_pages.empty()) {
            this->erase(this->entries.find(this->recency.back()));
        }
        const Key key{seq, page_ids[r]};
        Entry e;
        e.pool_page = this->pool->alloc_page();
        this->recency.push_front(key);
        e.recency = this->recency.begin();
        // The buckets are exclusive, so a scored set never names a page twice; a
        // duplicate would leak the slot it just claimed.
        TORCH_CHECK(
            this->entries.emplace(key, e).second,
            "RecallBuffer::begin_cycle: page ",
            page_ids[r],
            " appears twice in the scored set"
        );
        this->cycle_slots[r] = e.pool_page;
        this->cycle_miss[r] = 1;
    }
    this->n_miss = static_cast<int64_t>(miss.size());
    // Pool-page gather index for candidates(), built once for every layer.
    this->gather_index = at::tensor(this->cycle_slots, at::TensorOptions().dtype(at::kLong)).to(this->pool->device);
    return miss;
}

void RecallBuffer::fill(int64_t layer_index, const std::vector<int64_t>& rows, const at::Tensor& src) {
    const int64_t n = static_cast<int64_t>(rows.size());
    if (n == 0) {
        return;
    }
    TORCH_CHECK(src.size(0) == n, "RecallBuffer::fill: src has ", src.size(0), " pages, expected ", n);
    const int64_t staged = this->staged_pages();
    std::vector<int64_t> slots;
    slots.reserve(n);
    for (int64_t j = 0; j < n; ++j) {
        const int64_t r = rows[j];
        TORCH_CHECK(r >= 0 && r < staged, "RecallBuffer::fill: row ", r, " outside the cycle's scored set");
        TORCH_CHECK(
            this->cycle_miss[r],
            "RecallBuffer::fill: row ",
            r,
            " is already resident; only miss rows are staged"
        );
        slots.push_back(this->cycle_slots[r]);
    }
    at::Tensor idx = at::tensor(slots, at::TensorOptions().dtype(at::kLong)).to(this->pool->device);
    at::Tensor layer = this->pool->layer_k(layer_index);
    layer.index_copy_(0, idx, src.to(this->pool->device, this->pool->dtype).contiguous());
    this->cycle_filled[layer_index] += n;
}

at::Tensor RecallBuffer::candidates(int64_t layer_index) const {
    TORCH_CHECK(
        this->cycle_filled.at(layer_index) == this->n_miss,
        "RecallBuffer::candidates: layer ",
        layer_index,
        " has ",
        this->cycle_filled.at(layer_index),
        " filled rows, expected the cycle's ",
        this->n_miss,
        " miss rows"
    );
    // Cache residency is scattered, so scored order is a device gather.
    return this->pool->layer_k(layer_index).index_select(0, this->gather_index);
}

void RecallBuffer::drop(int64_t seq, int64_t page_id) {
    auto it = this->entries.find(Key{seq, page_id});
    if (it != this->entries.end()) {
        this->erase(it);
    }
}

void RecallBuffer::end_seq(int64_t seq) {
    for (auto it = this->entries.begin(); it != this->entries.end();) {
        if (it->first.seq != seq) {
            ++it;
            continue;
        }
        this->pool->free_page(it->second.pool_page);
        this->recency.erase(it->second.recency);
        it = this->entries.erase(it);
    }
}

}  // namespace pulsar
