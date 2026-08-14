#include "pulsar/runtime/kv/kv_memory.hpp"
#include "pulsar/profiling.hpp"

#include "host_pages.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// KVMemory routing: the evict/spill/recall cycle the read-layer header declares. The
// active buffer, the recall buffer and the buckets stay self-contained; this is the
// only file that references more than one of them at a time.

namespace pulsar {

namespace {

CascadeRetriever make_retriever(const RecallConfig& c) {
    return CascadeRetriever{.secondary = AttnScoreRetriever{c.scale, c.rope_theta, c.mass_reference_length}};
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

}  // namespace

const at::Tensor& SegmentDoc::document() {
    if (this->dirty) {
        std::vector<int64_t> t;
        t.reserve(this->ids.size());
        // A placeholder (an address inside the segment whose id was never recorded)
        // is not a term.
        for (int64_t x : this->ids) {
            if (x >= 0) {
                t.push_back(x);
            }
        }
        this->terms = at::tensor(t, at::TensorOptions().dtype(at::kLong));
        this->dirty = false;
    }
    return this->terms;
}

// Select the pages to promote. Scored pages are walked best first and each wins its own
// draw against the bar; a winner is promoted together with every candidate page within
// block_radius page ids of it. `pid` is ascending (the candidate list is page_id sorted),
// so a neighbourhood is a contiguous index range and the expansion is two binary searches.
std::vector<int64_t> select_recall_pages(
    const at::Tensor& scores,
    const std::vector<int64_t>& page_tokens,
    const std::vector<int64_t>& pid,
    int64_t block_radius,
    int64_t capacity_tokens,
    double bar,
    double temperature,
    std::mt19937_64& rng,
    std::vector<double>* selected_scores
) {
    std::vector<int64_t> sel;
    if (selected_scores) {
        selected_scores->clear();
    }
    if (capacity_tokens <= 0) {
        return sel;
    }
    at::Tensor scores_cpu = scores.to(at::kCPU, at::kDouble).contiguous();
    auto scores_a = scores_cpu.accessor<double, 1>();
    const int64_t n = scores_cpu.numel();
    TORCH_CHECK(
        static_cast<int64_t>(page_tokens.size()) == n,
        "select_recall_pages: ",
        page_tokens.size(),
        " page token counts for ",
        n,
        " scores"
    );
    TORCH_CHECK(
        static_cast<int64_t>(pid.size()) == n,
        "select_recall_pages: ",
        pid.size(),
        " page ids for ",
        n,
        " scores"
    );

    // Only a SCORED page may start a promotion: -inf marks a page whose segment lost the
    // prefilter, and NaN is not a score. Such a page is still reachable through another
    // page's expansion -- it is recalled as context, not on its own merit. A candidate
    // whose r overflowed is the strongest one there is, not the weakest.
    std::vector<int64_t> order;
    order.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isnan(scores_a[i]) && scores_a[i] != -std::numeric_limits<double>::infinity()) {
            order.push_back(i);
        }
    }
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        if (scores_a[a] != scores_a[b]) {
            return scores_a[a] > scores_a[b];
        }
        return a < b;
    });

    // `bar` is the eviction keep-bar itself, and `scores` are would-be densities, so this
    // is the same test on the same quantity in both directions.
    const double log_bar = std::log(bar);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<char> taken(n, 0);
    int64_t promoted_tokens = 0;
    for (int64_t i : order) {
        if (taken[i]) {
            continue;  // already pulled in by an earlier page's expansion
        }
        const double density = scores_a[i];
        double keep_probability;
        if (bar <= 0.0) {
            keep_probability = 1.0;  // bar at 0: every candidate passes
        } else if (!std::isfinite(bar)) {
            keep_probability = 0.0;  // bar at +inf: none does
        } else if (temperature <= 0.0) {
            keep_probability = density >= bar ? 1.0 : 0.0;  // hard step at the bar
        } else if (density <= 0.0) {
            keep_probability = 0.0;  // log density is -inf here
        } else {
            // Softened in log-density, the score's own log-odds.
            keep_probability = sigmoid((std::log(density) - log_bar) / temperature);
        }
        // Only a probability strictly inside (0, 1) costs a draw.
        const bool keep = keep_probability >= 1.0 ? true
                                                  : (keep_probability <= 0.0 ? false : unif(rng) < keep_probability);
        if (!keep) {
            continue;
        }
        // The winner's neighbourhood in page_id space. A page id with no candidate (still
        // resident, or in neither bucket) simply is not in the range.
        const int64_t lo = std::lower_bound(pid.begin(), pid.end(), pid[i] - block_radius) - pid.begin();
        const int64_t hi = std::upper_bound(pid.begin(), pid.end(), pid[i] + block_radius) - pid.begin();
        int64_t cost = 0;
        for (int64_t q = lo; q < hi; ++q) {
            if (!taken[q]) {
                cost += page_tokens[q];
            }
        }
        // A neighbourhood that does not fit is skipped whole, and the walk continues: the
        // promotion is the neighbourhood, so admitting part of it is not the same request.
        // A later, smaller one may still fit.
        if (promoted_tokens + cost > capacity_tokens) {
            continue;
        }
        // The whole neighbourhood, the winner included, is promoted on the WINNER's score.
        for (int64_t q = lo; q < hi; ++q) {
            if (taken[q]) {
                continue;
            }
            taken[q] = 1;
            sel.push_back(q);
            if (selected_scores) {
                selected_scores->push_back(density);
            }
        }
        promoted_tokens += cost;
    }
    return sel;
}

KVMemory::KVMemory(
    ActiveBuffer& active,
    RecallBuffer& recall_buffer,
    PagedBucket& s1,
    FileBucket& s2,
    RecallConfig config,
    bool skip_fresh
)
    : active_buf(active),
      recall_buffer(recall_buffer),
      s1(s1),
      s2(s2),
      config(std::move(config)),
      skip_fresh(skip_fresh),
      retriever(make_retriever(this->config)),
      rng(static_cast<std::mt19937_64::result_type>(this->config.seed >= 0 ? this->config.seed : 0)) {
    TORCH_CHECK(!this->config.layers.empty(), "KVMemory: cascade reranker needs at least one layer");
}

void KVMemory::accept_evicted(int64_t seq, const DemotedKV& demoted, int64_t step) {
    if (demoted.num_tokens() == 0) {
        return;
    }
    Reservation r = this->s1.reserve(seq, demoted, step);
    this->s1.fill(seq, this->active_buf, r);
    std::unordered_set<int64_t>& f = this->fresh[seq];
    for (int64_t pid : r.page_ids) {
        f.insert(pid);
    }
}

void KVMemory::end_cycle(int64_t seq, int64_t ram_bucket_size, int64_t step, double decay) {
    // The spill reads s1's KV, and the next cycle's eviction overwrites the victim
    // slots this cycle's fill read from.
    this->s1.wait_fill();
    this->s1.release_pending(seq);
    this->s2.release_pending(seq);
    if (this->s2.enabled() && ram_bucket_size > 0) {
        for (PageKV& pg : this->s1.spill_to_budget(seq, ram_bucket_size, step, decay)) {
            this->s2.accept(seq, pg);
        }
        this->s1.release_pending(seq);
    }
    this->fresh.erase(seq);
}

void KVMemory::end_seq(int64_t seq) {
    this->fresh.erase(seq);
    this->segments.erase(seq);
    this->recall_buffer.end_seq(seq);
}

void KVMemory::note_tokens(
    int64_t seq,
    const std::vector<int64_t>& addr,
    const std::vector<int64_t>& toks,
    int64_t segment_size
) {
    TORCH_CHECK(addr.size() == toks.size(), "note_tokens: addr/token length mismatch");
    const int64_t page_size = this->active_buf.page_size();
    TORCH_CHECK(
        segment_size > 0 && segment_size % page_size == 0,
        "note_tokens: segment_size ",
        segment_size,
        " must be a positive multiple of page_size ",
        page_size
    );
    SegmentIndex& index = this->segments[seq];
    if (index.segment_size == 0) {
        index.segment_size = segment_size;
    }
    // The segment id IS an address divided by this, so a sequence whose grouping
    // changed would file later ids under keys the earlier ones do not share.
    TORCH_CHECK(
        index.segment_size == segment_size,
        "note_tokens: seq ",
        seq,
        " records its documents at segment_size ",
        index.segment_size,
        "; it cannot change to ",
        segment_size
    );
    std::map<int64_t, SegmentDoc>& docs = index.docs;
    for (size_t i = 0; i < addr.size(); ++i) {
        const int64_t a = addr[i];
        if (a < 0) {
            continue;
        }
        SegmentDoc& d = docs[a / segment_size];
        const int64_t off = a % segment_size;
        if (off >= static_cast<int64_t>(d.ids.size())) {
            d.ids.resize(off + 1, -1);
        }
        d.ids[off] = toks[i];
        d.dirty = true;
    }
}

std::vector<int64_t> KVMemory::token_ids_in(int64_t seq, int64_t start, int64_t end) const {
    std::vector<int64_t> out;
    if (end <= start) {
        return out;
    }
    auto seq_it = this->segments.find(seq);
    TORCH_CHECK(seq_it != this->segments.end(), "token_ids_in: seq ", seq, " has no recorded documents");
    const SegmentIndex& index = seq_it->second;
    const int64_t ss = index.segment_size;
    TORCH_CHECK(ss > 0, "token_ids_in: seq ", seq, " has no segment size");
    out.reserve(end - start);
    for (int64_t a = start; a < end;) {
        const int64_t sid = a / ss;
        auto doc = index.docs.find(sid);
        TORCH_CHECK(
            doc != index.docs.end(),
            "token_ids_in: seq ",
            seq,
            " has no document for segment ",
            sid,
            " (address ",
            a,
            ")"
        );
        const std::vector<int64_t>& ids = doc->second.ids;
        const int64_t seg_end = std::min(end, (sid + 1) * ss);
        for (; a < seg_end; ++a) {
            const int64_t off = a - sid * ss;
            TORCH_CHECK(
                off < std::ssize(ids) && ids[off] >= 0,
                "token_ids_in: address ",
                a,
                " was never recorded for seq ",
                seq
            );
            out.push_back(ids[off]);
        }
    }
    return out;
}

void KVMemory::partition_by_bucket(
    const std::vector<Cand>& pages,
    const std::vector<int64_t>& scored_pages,
    const std::vector<int64_t>& rows,
    std::vector<std::vector<int64_t>>& by_rows,
    std::vector<std::vector<int64_t>>& by_pids
) {
    by_rows.assign(2, {});
    by_pids.assign(2, {});
    for (int64_t j : rows) {
        const Cand& c = pages[scored_pages[j]];
        by_rows[c.bucket].push_back(j);
        by_pids[c.bucket].push_back(c.page_id);
    }
}

std::vector<at::Tensor> KVMemory::stage_candidates(
    int64_t seq,
    const std::vector<Cand>& pages,
    const std::vector<int64_t>& scored_pages,
    RecallCounts* counts
) {
    PULSAR_PROF_SCOPE(RERANK_STAGE);
    const int64_t n_qk = static_cast<int64_t>(this->config.layers.size());
    const int64_t n_scored = static_cast<int64_t>(scored_pages.size());
    std::vector<int64_t> scored_ids(n_scored);
    for (int64_t j = 0; j < n_scored; ++j) {
        scored_ids[j] = pages[scored_pages[j]].page_id;
    }
    // The whole scored set must be simultaneously resident, so a cycle that would
    // overrun the preallocated pool is a hard error, not a clamp and not a VRAM grow.
    // Only the rows the cache misses cost a bucket read.
    std::vector<int64_t> miss = this->recall_buffer.begin_cycle(seq, scored_ids);
    if (counts) {
        counts->staged_misses += static_cast<int64_t>(miss.size());
        counts->staged_hits += n_scored - static_cast<int64_t>(miss.size());
    }
    std::vector<at::Tensor> cand_k(n_qk);
    std::vector<std::vector<int64_t>> rows, pids;
    KVMemory::partition_by_bucket(pages, scored_pages, miss, rows, pids);
    GatheredK s1_k, s2_k;
    if (!pids[0].empty()) {
        s1_k = this->s1.gather_k(seq, pids[0], this->config.layers);
    }
    if (!pids[1].empty()) {
        s2_k = this->s2.gather_k(seq, pids[1], this->config.layers);
    }
    // A bucket gathers in its own storage order, so the staging rows follow its row
    // order, not the order the pages were asked for.
    auto reorder = [](const std::vector<int64_t>& src, const std::vector<int64_t>& order) {
        std::vector<int64_t> out;
        out.reserve(order.size());
        for (int64_t i : order) {
            out.push_back(src[i]);
        }
        return out;
    };
    const std::vector<int64_t> s1_rows = reorder(rows[0], s1_k.order);
    const std::vector<int64_t> s2_rows = reorder(rows[1], s2_k.order);
    for (int64_t layer_index = 0; layer_index < n_qk; ++layer_index) {
        if (!s1_k.layers.empty()) {
            this->recall_buffer.fill(layer_index, s1_rows, s1_k.layers[layer_index]);
        }
        if (!s2_k.layers.empty()) {
            this->recall_buffer.fill(layer_index, s2_rows, s2_k.layers[layer_index]);
        }
        cand_k[layer_index] = this->recall_buffer.candidates(layer_index);
    }
    return cand_k;
}

at::Tensor KVMemory::rerank_gpu(
    const std::vector<at::Tensor>& query_vecs,
    const std::vector<at::Tensor>& cand_k,
    const std::vector<at::Tensor>& lse,
    const at::Tensor& log_attended_len,
    const at::Tensor& query_positions,
    int64_t stored_key_position,
    int64_t scored_key_position
) {
    const int64_t n_scored = cand_k.empty() ? 0 : cand_k[0].size(0);
    if (n_scored == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kDouble));
    }
    return this->retriever.secondary
        .rerank_score_batched(
            query_vecs,
            cand_k,
            lse,
            log_attended_len,
            query_positions,
            stored_key_position,
            scored_key_position
        )
        .to(at::kCPU, at::kDouble)
        .contiguous();
}

std::vector<Payload> KVMemory::recall(
    int64_t seq,
    const at::Tensor& query_token_ids,
    int64_t capacity_tokens,
    const std::vector<at::Tensor>& query_vecs,
    const std::vector<at::Tensor>& lse,
    const at::Tensor& log_attended_len,
    const at::Tensor& query_positions,
    const RecallParams& params,
    std::vector<double>* raw_scores,
    RecallCounts* counts
) {
    std::vector<Payload> out;
    // The captured denominator is the score's reference; a cycle without one has
    // nothing to score against.
    TORCH_CHECK(!lse.empty() && log_attended_len.defined(), "recall: needs the captured lse and attended lengths");
    TORCH_CHECK(query_positions.defined(), "recall: needs the query rows' buffer positions");
    TORCH_CHECK(
        params.short_offset > 0,
        "recall: short_offset (",
        params.short_offset,
        ") must be a resolved absolute position > 0"
    );
    TORCH_CHECK(
        params.candidate_position >= 0,
        "recall: candidate_position (",
        params.candidate_position,
        ") must not be negative"
    );
    if (capacity_tokens <= 0) {
        return out;
    }
    const int64_t page_size = this->active_buf.page_size();
    TORCH_CHECK(
        params.segment_size > 0 && params.segment_size % page_size == 0,
        "recall: segment_size ",
        params.segment_size,
        " must be a positive multiple of page_size ",
        page_size
    );
    const int64_t sps = params.segment_size / page_size;  // pages per segment
    const int64_t n_layers = this->active_buf.n_layers();
    const int64_t n_qk = static_cast<int64_t>(this->config.layers.size());
    const double neg_inf = -std::numeric_limits<double>::infinity();

    // Candidate pages across the bucket chain (the buckets are exclusive, so the
    // merge of their cursors is the demoted set). token_ids stay as host vectors in
    // the buckets' RAM indices; no per-page tensor is built here. The cursors must
    // outlive `pages`, and no take_pages may run until every read of them is done.
    BucketView v1 = this->s1.view(seq);
    BucketView v2 = this->s2.view(seq);
    const std::unordered_set<int64_t>* fresh_set = nullptr;
    if (this->skip_fresh) {
        auto fit = this->fresh.find(seq);
        if (fit != this->fresh.end()) {
            fresh_set = &fit->second;
        }
    }
    auto selectable_page = [&](int64_t pid) { return fresh_set == nullptr || fresh_set->count(pid) == 0; };
    std::vector<Cand> pages;
    pages.reserve(v1.pages.size() + v2.pages.size());
    for (const BucketPageRef& p : v1.pages) {
        if (selectable_page(p.page_id)) {
            pages.push_back(Cand{0, p.page_id, p.token_ids});
        }
    }
    for (const BucketPageRef& p : v2.pages) {
        if (selectable_page(p.page_id)) {
            pages.push_back(Cand{1, p.page_id, p.token_ids});
        }
    }
    if (pages.empty()) {
        return out;
    }
    int64_t smallest_page = std::numeric_limits<int64_t>::max();
    for (const Cand& c : pages) {
        smallest_page = std::min<int64_t>(smallest_page, static_cast<int64_t>(c.token_ids->size()));
    }
    if (smallest_page > capacity_tokens) {
        return out;
    }
    // Both cursors are ascending and the buckets are disjoint; sort the union by
    // page index so the page list is in address order and each segment is a
    // contiguous index range.
    std::sort(pages.begin(), pages.end(), [](const Cand& a, const Cand& b) { return a.page_id < b.page_id; });
    const int64_t n_pages = static_cast<int64_t>(pages.size());

    // BM25 documents are the residency-independent segment corpus: a document is
    // the token ids at a segment's addresses, frozen once the segment fills, so it
    // describes what was SAID rather than what happens to be demoted. That makes idf
    // the statistic over the whole conversation and scores comparable across cycles.
    // SELECTION restricts to segments holding at least one candidate page: a
    // fully-active segment can score high and contribute nothing, so it is skipped
    // rather than consuming a prefilter slot. page_seg maps a candidate page to its
    // document row; only kept-segment pages are scored, so a page whose segment loses the
    // prefilter ends at -inf and can only enter through a winner's expansion.
    auto seg_it = this->segments.find(seq);
    TORCH_CHECK(
        seg_it != this->segments.end(),
        "recall: seq ",
        seq,
        " has no segment documents (note_tokens was "
        "never called for it)"
    );
    // A page's document row is looked up by page_id / sps, so scoring under a grouping
    // the documents were not recorded under would read the wrong document.
    TORCH_CHECK(
        seg_it->second.segment_size == params.segment_size,
        "recall: seq ",
        seq,
        " recorded its documents at segment_size ",
        seg_it->second.segment_size,
        "; this cycle asks for ",
        params.segment_size
    );
    std::vector<RetrievalDoc> seg_docs;
    std::unordered_map<int64_t, int64_t> seg_row;  // segment id -> document row
    seg_docs.reserve(seg_it->second.docs.size());
    for (auto& [sid, doc] : seg_it->second.docs) {
        RetrievalDoc rd;
        rd.token_ids = doc.document();  // cached; only a partial segment rebuilds
        seg_row.emplace(sid, static_cast<int64_t>(seg_docs.size()));
        seg_docs.push_back(std::move(rd));
    }
    const int64_t n_segs = static_cast<int64_t>(seg_docs.size());
    std::vector<char> selectable(n_segs, 0);
    std::vector<int64_t> page_seg(n_pages);
    for (int64_t p = 0; p < n_pages; ++p) {
        auto row = seg_row.find(pages[p].page_id / sps);
        TORCH_CHECK(
            row != seg_row.end(),
            "recall: page ",
            pages[p].page_id,
            " has no segment document (note_tokens missed its addresses)"
        );
        page_seg[p] = row->second;
        selectable[row->second] = 1;
    }

    TORCH_CHECK(
        static_cast<int64_t>(query_vecs.size()) == n_qk,
        "recall: query has ",
        query_vecs.size(),
        " layer vectors, expected ",
        n_qk
    );
    const RetrievalDoc query{query_token_ids, {}};  // prefilter reads token_ids only

    // PHASE 1 -- prefilter: rank all candidate segments (BM25 documents) by the cascade's
    // BM25 leg on token_ids only (all in RAM); take the top prefilter_k winners.
    std::vector<int64_t> kept = this->retriever.prefilter(query, seg_docs, params.prefilter_k, selectable);
    std::vector<char> seg_kept(n_segs, 0);
    for (int64_t s : kept) {
        seg_kept[s] = 1;
    }

    // PHASE 2 -- rerank: score each kept-segment page. Bucket gathers are bounded to the
    // scored pages. Nothing outside a kept segment is scored: a neighbour reaches the
    // active buffer through another page's expansion in selection, which needs its
    // membership and its token count, not a score.
    // page_id per candidate (pages ascending by page_id). Expansion works in page_id
    // distance, so an attended-context gap wider than the radius breaks the neighbourhood
    // by distance alone (no explicit run split).
    std::vector<int64_t> pid(n_pages);
    // Token count per candidate page: what selection charges against its capacity. Read
    // from the bucket's RAM token ids, so it must be taken before any take_pages.
    std::vector<int64_t> page_tokens(n_pages);
    for (int64_t p = 0; p < n_pages; ++p) {
        pid[p] = pages[p].page_id;
        page_tokens[p] = static_cast<int64_t>(pages[p].token_ids->size());
    }
    std::vector<int64_t> scored_pages;  // indices into `pages` that get a rerank score
    scored_pages.reserve(n_pages);
    for (int64_t p = 0; p < n_pages; ++p) {
        if (seg_kept[page_seg[p]]) {
            scored_pages.push_back(p);
        }
    }
    const int64_t n_scored = static_cast<int64_t>(scored_pages.size());

    // First read of bucket KV this cycle: the eviction's transfers ran under the
    // prefilter above and are fenced here.
    this->s1.wait_fill();
    // The scored pages' relevance-layer K is staged in the device RecallBuffer (misses
    // only), then every page is scored in one batched GPU matmul against the softmax
    // denominator the attention itself used for each query row.
    std::vector<at::Tensor> cand_k = this->stage_candidates(seq, pages, scored_pages, counts);
    at::Tensor sec = PULSAR_PROF_EXPR(
        RERANK,
        this->rerank_gpu(
            query_vecs,
            cand_k,
            lse,
            log_attended_len,
            query_positions,
            params.short_offset,
            params.candidate_position
        )
    );

    // Scatter the scored pages' raw rerank scores back to the full page list; a page
    // outside a kept segment stays -inf and cannot start a promotion.
    at::Tensor sec_cpu = sec.to(at::kCPU, at::kDouble).contiguous();
    auto sec_a = sec_cpu.accessor<double, 1>();
    if (raw_scores) {
        raw_scores->resize(n_scored);
        for (int64_t j = 0; j < n_scored; ++j) {
            (*raw_scores)[j] = sec_a[j];
        }
    }
    std::vector<double> raw(n_pages, neg_inf);
    for (int64_t j = 0; j < n_scored; ++j) {
        raw[scored_pages[j]] = sec_a[j];
    }
    at::Tensor scores = at::tensor(raw, at::TensorOptions().dtype(at::kDouble));

    // Selection: walk the scored pages best first and promote each by its own Bernoulli
    // draw against the keep-bar, taking the winner's whole page_id neighbourhood with it
    // and stopping at capacity_tokens.
    std::vector<double> sel_score;
    std::vector<int64_t> sel = select_recall_pages(
        scores,
        page_tokens,
        pid,
        params.block_radius,
        capacity_tokens,
        params.bar,
        params.temperature,
        this->rng,
        &sel_score
    );
    if (sel.empty()) {
        return out;
    }

    // Promote: remove the winners from the bucket that holds them and read their full
    // KV out of it, one batched take per bucket. Nothing may read `pages`' token_ids
    // pointers past this point.
    std::vector<std::vector<int64_t>> take(2);
    for (int64_t sel_idx : sel) {
        take[pages[sel_idx].bucket].push_back(pages[sel_idx].page_id);
    }
    if (counts) {
        counts->promoted_s1 += static_cast<int64_t>(take[0].size());
        counts->promoted_s2 += static_cast<int64_t>(take[1].size());
    }
    std::unordered_map<int64_t, PageKV> promoted;
    for (PageKV& pg : this->s1.take_pages(seq, take[0])) {
        promoted.emplace(pg.page_id, std::move(pg));
    }
    for (PageKV& pg : this->s2.take_pages(seq, take[1])) {
        promoted.emplace(pg.page_id, std::move(pg));
    }
    // A promotion takes the page out of the bucket chain, the one event that
    // invalidates a cached copy. A spill between buckets does not: demoted K is
    // stored roped at the session's short_offset and never modified while demoted.
    for (const std::vector<int64_t>& ids : take) {
        for (int64_t page_id : ids) {
            this->recall_buffer.drop(seq, page_id);
        }
    }

    // One Payload per promoted page, in selection order (each winner followed by its
    // neighbourhood). recompact_with sorts the merged set by address, so this order is
    // not load-bearing.
    out.reserve(sel.size());
    for (size_t rank = 0; rank < sel.size(); ++rank) {
        const int64_t sel_idx = sel[rank];
        PageKV& pg = promoted.at(pages[sel_idx].page_id);
        Payload payload;
        payload.score = sel_score[rank];
        payload.token_ids = std::move(pg.token_ids);
        payload.k = std::move(pg.k);  // [cnt, n_layers, n_kv_heads, head_dim]
        payload.v = std::move(pg.v);
        TORCH_CHECK(
            payload.k.size(1) == n_layers,
            "recall: promoted page has ",
            payload.k.size(1),
            " layers, expected ",
            n_layers
        );
        payload.base_addr = pages[sel_idx].page_id * page_size;
        out.push_back(std::move(payload));
    }
    return out;
}

}  // namespace pulsar
