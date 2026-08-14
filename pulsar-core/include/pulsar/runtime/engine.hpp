#pragma once

#include "pulsar/runtime/kv/active_buffer.hpp"
#include "pulsar/runtime/kv/file_bucket.hpp"
#include "pulsar/runtime/kv/kv_memory.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"
#include "pulsar/runtime/kv/recall.hpp"
#include "pulsar/runtime/sampler.hpp"
#include "pulsar/runtime/scheduler.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Generator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Continuous-batching generation over a Model, driving the Scheduler and ActiveBuffer.
// feed() enqueues tokens onto a sequence; step(n) runs scheduler iterations (build
// groups -> model.forward -> sample -> update) until n tokens are emitted, prefilling to
// completion as needed. A finished sequence is PARKED, keeping its KV, so a later feed()
// continues it and release() frees it.
//
// Plain C++ class; the Python-facing surface is the EngineBinding shim
// (pulsar/ext/engine.cpp). Owns its Model, ActiveBuffer, Scheduler, Sampler, and, when
// recall is on, the RecallBuffer, the s1/s2 spill buckets and the KVMemory.

namespace pulsar {

// A keep-bar on one quantity, the attention share a page holds as a multiple of UNIFORM
// over the length that share is stated against (see EngineConfig::mass_reference_length):
// a page leaves when the share it receives falls below the EVICTION bar and returns when
// the share it WOULD receive rises above the RECALL bar. tau is r / (1 + r) for r that
// multiple, so the bar is r itself and 1.0 is parity with uniform at that length.
//
// tau <= 0 bars nothing and tau >= 1 bars everything. Each direction is built from its
// own tau and neither value moves the other: evict_tau > 0 with recall_tau >= 1 is
// eviction only, and evict_tau 0 with recall_tau > 0 leaves capacity the only thing that
// demotes.
inline double evict_bar_from_tau(double tau) {
    if (tau <= 0.0) {
        return 0.0;
    }
    if (tau >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return tau / (1.0 - tau);
}

// The per-token attention-mass EMA rate alpha, against the engine's active_max_size.
// <= 0 derives 1 / active_max_size, which is also the FLOOR the result is clamped up
// to. Capped at 1 (alpha == 1 keeps only the current step). An active_max_size <= 0
// returns 0.
inline double resolve_mass_decay(double configured, int64_t active_max_size) {
    if (active_max_size <= 0) {
        return 0.0;
    }
    const double floor_rate = 1.0 / static_cast<double>(active_max_size);
    const double alpha = configured > 0.0 ? configured : floor_rate;
    return std::min(std::max(alpha, floor_rate), 1.0);
}

// The neighbourhood radius is configured in TOKENS and both sites work in PAGES. A radius
// that is not a whole number of pages is rejected rather than rounded, so the radius that
// runs is the one configured.
inline int64_t resolve_block_radius(int64_t radius_tokens, int64_t page_size) {
    TORCH_CHECK(radius_tokens >= 0, "block radius ", radius_tokens, " must not be negative");
    TORCH_CHECK(
        radius_tokens % page_size == 0,
        "block radius ",
        radius_tokens,
        " tokens is not a multiple of the ",
        page_size,
        "-token page"
    );
    return radius_tokens / page_size;
}

// The decode cadence in TOKENS: generated tokens between a sequence's evict->recall
// cycles. <= 0 takes max_chunk_size, the prefill chunk budget, so a configuration that
// names neither still cycles.
inline int64_t resolve_decode_cycle_interval(int64_t configured, int64_t max_chunk_size) {
    const int64_t interval = configured > 0 ? configured : max_chunk_size;
    TORCH_CHECK(
        interval > 0,
        "decode_cycle_interval (",
        configured,
        ") resolves to ",
        interval,
        " tokens; it must be > 0"
    );
    return interval;
}

// The BM25 prefilter's QUERY length in TOKENS. <= 0 takes n_working, the whole protected
// window. Independent of the rerank query width (recall_query_window) and of what
// eviction protects: this decides only how many recent token ids become BM25 terms.
inline int64_t resolve_bm25_query_tokens(int64_t configured, int64_t n_working) {
    const int64_t tokens = configured > 0 ? configured : n_working;
    TORCH_CHECK(
        tokens > 0,
        "recall_bm25_query_tokens (",
        configured,
        ") resolves to ",
        tokens,
        " tokens; it must be > 0"
    );
    return tokens;
}

// Clamped to the stream start, so the query is shorter than `tokens` only at the head
// of a stream.
inline int64_t bm25_query_start(int64_t end, int64_t tokens) {
    return std::max<int64_t>(0, end - tokens);
}

// The knobs a SESSION owns, as configured: what create()'s override dict may set and
// what EngineConfig's matching fields seed. Everything here is read by the per-sequence
// evict->recall cycle, which runs one sequence at a time after the batched forward, so
// two sessions may differ on any of it without touching batching.
//
// What is NOT here is engine-level for a reason: pool geometry (active_buffer_size,
// page_size, max_sessions, max_running, max_chunk_size) sizes one shared pool; the
// bucket budgets size the buckets; relevance_layers is read INSIDE the batched forward
// (set_capture_layers is model-global), so a per-session value would need a per-sequence
// kernel argument.
struct SessionConfig {
    // Protected recent window, never evicted. Also the default length of the BM25
    // prefilter's query (see recall_bm25_query_tokens). NO default: a session that
    // recalls must state it, because it is the one knob whose right value follows from
    // the workload rather than from the pool geometry, and a derived one would be a
    // guess that never announces itself. Unread with recall off.
    int64_t n_working = 0;
    // Recent tokens whose Q forms the RERANK query, in TOKENS. Must be > 0; 1 is the most
    // recent token alone. Independent of n_working: the protected window is a
    // retention decision and this is a query width. It does NOT govern the BM25
    // prefilter's query, which is recall_bm25_query_tokens long.
    int64_t recall_query_window = 1;
    // Neighbourhood radius in TOKENS, 0 = off, a whole number of pages. Fragmentation
    // control over the demoted/promoted set, not part of the score, and the same radius
    // in both directions: eviction keeps a page whose neighbourhood clears the bar, and
    // recall promotes a winning page's whole neighbourhood. At the default 256 with a
    // 16-token page the window is 2*16+1 = 33 pages, so one page above the bar carries
    // 528 tokens.
    int64_t block_radius = 256;
    // How the active buffer's slot indices map to RoPE positions: "contiguous" (the slot
    // index IS the position) or "compacted" (sink, collapsed distant region, working
    // window -- the InfLLM/LM-Infinite layout). Parsed once at create().
    std::string active_position_layout = "contiguous";
    // The absolute RoPE position, in TOKENS, that demoted keys are roped to and that the
    // compacted layout collapses its distant region onto. 0 is a sentinel for n_sink,
    // resolved at the feed that sets the sink. A non-zero value must be >= this
    // session's n_sink. Read under BOTH layouts: under "contiguous" it sets the demotion
    // rotation without collapsing the buffer.
    int64_t short_offset = 0;
    int64_t recall_segment_size = 0;  // tokens, a multiple of page_size; <= 0 => page_size
    int64_t recall_prefilter_k = 24;  // BM25 prefilter width M; <= 0 scores all segments
    // The BM25 prefilter's query length in TOKENS, taken off the end of the stream;
    // <= 0 => n_working. Separate from n_working because the protected window is a
    // retention decision and this is a query width.
    int64_t recall_bm25_query_tokens = 0;
    // tau for the RECALL bar, in [0, 1]: selection compares each candidate's would-be
    // density against evict_bar_from_tau(this). 0 recalls every scored page.
    double recall_tau = 0.0;
    // tau for the EVICTION bar, same units and same formula, independent of recall_tau:
    // the density pass demotes a page whose measured density falls below
    // evict_bar_from_tau(this). 0 demotes nothing, leaving the capacity backstop.
    double evict_tau = 0.0;
    // Softens the bar: 0 is a hard step, > 0 promotes each page by an independent
    // Bernoulli draw on its log-odds distance from the bar, taken in descending score
    // until the cycle's token capacity is full. Larger flattens toward a coin flip.
    double recall_temperature = 0.0;
    // Per-TOKEN attention-mass forgetting rate alpha, in (0, 1]: the attention kernels
    // scale each query's mass contribution by it and the upkeep retains 1 - alpha per
    // token, so the EMA memory length is 1/alpha TOKENS. <= 0 derives
    // 1 / active_max_size, which is also the floor (see resolve_mass_decay). Read only
    // when recall is on; with recall off nothing accumulates or evicts. The kernels take
    // one alpha per sequence, so sessions in one batch may differ.
    double attention_mass_decay = 0.0;
    // Whether RECALL runs during PREFILL. Eviction is unaffected; the density bar applies
    // in both phases.
    //
    // A prefill cycle runs after its chunk is already resident, so it retrieves against
    // the chunk consumed rather than the one arriving, and cannot select for a question
    // that arrives last.
    bool recall_during_prefill = false;
};

// A session's knobs RESOLVED, fixed for its life at create(): defaults applied, tokens
// converted to the unit each site works in, tau converted to a density bar, and the
// head-reduce string parsed. Resolving once means a malformed value fails at create()
// rather than mid-run, and no cycle re-parses a string.
struct SessionRuntime {
    int64_t n_working = 0;
    int64_t query_window = 1;
    int64_t block_radius_pages = 0;
    PositionLayout position_layout = PositionLayout::contiguous;
    // As CONFIGURED: 0 is still the n_sink sentinel. SeqState::short_offset holds the
    // resolved value.
    int64_t short_offset = 0;
    int64_t segment_size = 0;
    int64_t prefilter_k = -1;
    int64_t bm25_query_tokens = 0;  // resolved; 0 iff recall is off
    double evict_bar = 0.0;  // what eviction demotes below
    double recall_bar = 0.0;  // what recall promotes above
    double recall_temperature = 0.0;
    double attention_mass_decay = 0.0;  // EMA rate alpha, resolved; 0 iff recall is off
    bool recall_during_prefill = false;
};

// Everything the Engine needs to build itself.
struct EngineConfig {
    // Model id: a family tag the factory parses from the prefix ("qwen3*" ->
    // Qwen3Model, "qwen2*" -> Qwen2Model, exactly "compiled" -> the AOTInductor .pt2
    // path). A trailing "-fp8"/"-int4" suffix is rejected; only bf16 is built. E.g.
    // "qwen2", "qwen3-0.6b", "compiled".
    std::string model;

    // native path: named-weight dict (already on the pool device/dtype) + the
    // full model dims. The pool geometry (n_layers/n_kv_heads/head_dim/rope_theta)
    // is read from these for BOTH model kinds; the compiled path ignores the
    // native-only dims (n_heads/hidden/intermediate/vocab/rms_eps), which are baked
    // in the .pt2.
    c10::Dict<std::string, at::Tensor> weights;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t n_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t hidden = 0;
    int64_t intermediate = 0;
    int64_t vocab = 0;
    double rope_theta = 0.0;
    double rms_eps = 0.0;

    // compiled path: the decode/prefill package paths.
    std::string decode_pt2;
    std::string prefill_pt2;

    // The length in TOKENS the eviction/recall density is stated against: a key holding
    // a uniform share of a context this long scores exactly 1, parity with the bar. <= 0
    // states the density against each query's OWN attended key count, under which
    // uniform attention scores 1 at every length. Engine-level: it must be the same on
    // both sides of the bar.
    int64_t mass_reference_length = 0;

    // Sessions that may hold KV at once. Sizes the ActiveBuffer pool
    // (max_sessions * active_buffer_size / page_size pages) and the compiled path's
    // static batch width. max_running cannot own this: a parked session holds KV
    // while not running, so capacity must cover running PLUS parked, and
    // max_running > max_sessions is a construction error.
    int64_t max_sessions = 1;

    // Attended-KV sizing. active_buffer_size is the per-sequence ActiveBuffer in
    // TOKENS: the window the model attends over, the unit the trained context length
    // is expressed in. It sizes the physical pool exactly (max_sessions *
    // active_buffer_size / page_size pages) and is the hard cap a sequence's active
    // length may never pass, above the soft target Engine::active_max_size drives to.
    // Must be a multiple of page_size.
    int64_t active_buffer_size = 0;
    int64_t page_size = 16;  // the tensor-core paged kernels take 16 or 32
    at::ScalarType dtype = at::ScalarType::Undefined;
    std::string device;

    // Scheduler.
    int64_t max_running = 0;
    int64_t max_chunk_size = 0;

    // Generation / sampler defaults. A sequence stops on eos_id (-1: none, runs as
    // long as the caller keeps stepping); the rest are GenerationParams.
    int64_t eos_id = -1;
    double temperature = 0.0;
    double top_p = 1.0;
    int64_t top_k = 0;
    int64_t seed = -1;

    // Recall. recall_buffer_size <= 0 keeps the non-recall path: no demoted KV, and no
    // eviction either, since one bar governs both directions.

    // The per-session knobs' DEFAULTS: what a session gets when create() names no
    // override. Dict keys are flat (n_working, block_radius, ...), so this nesting
    // is invisible to the config dict. See SessionConfig.
    SessionConfig session;

    // The top-most layers whose attention decides BOTH what to evict (the density's
    // layer subset) and what to recall (the reranker's scored layers). Empty defaults to
    // the top quarter [3 * n_layers / 4, n_layers). Engine-level: set_capture_layers is
    // model-global and the capture happens inside the batched forward.
    std::vector<int64_t> relevance_layers;

    // Per-seq CANDIDATE STAGING budget in TOKENS. The RecallBuffer pool is max_sessions
    // of these, so every live session's cycle stages at once (the same guarantee
    // ActiveBuffer gives with max_sessions whole windows). A cycle's whole scored set
    // must be simultaneously resident, so this is a hard FLOOR at the worst-case scored
    // set (recall_prefilter_k * recall_segment_size), NOT checked at construction: a
    // cycle that overruns it is a hard error at RecallBuffer::begin_cycle.
    // <= 0 turns recall off entirely.
    int64_t recall_buffer_size = 0;
    // S1 token budget. Above it the coldest pages spill to S2, so a budget requires a
    // spill_dir to drain into. <= 0 leaves S1 unbudgeted.
    int64_t ram_bucket_size = 0;
    // Parent directory for S2's per-sequence spill files. Empty disables S2 entirely and
    // creates no file.
    std::string spill_dir;
    // Filter pages evicted in the current cycle out of that cycle's candidate set. This
    // is what keeps the rerank off pages whose copy into S1 may still be in flight.
    bool recall_skip_fresh = true;
    // Generated tokens between the decode-cadence evict->recall cycles of a sequence that
    // is not prefilling, in TOKENS. <= 0 takes max_chunk_size; the resolved value must be
    // > 0 (see resolve_decode_cycle_interval).
    int64_t decode_cycle_interval = 0;
};

// One token emitted by a step: the sequence it belongs to, the token id, and
// whether that emission finished the sequence. step() returns one per token
// emitted that step (prefill group then decode group).
struct TokenEvent {
    int64_t seq_id;
    int64_t token;
    bool finished;
};

// Log10 bins of an eviction candidate's bias-corrected density, a multiple of uniform
// attention. The range must bracket parity with uniform, 1.0, since the eviction bar is
// stated in the same units. EngineBinding::stats publishes these edges and consumers
// label their axes from them, so they are declared once here.
struct DensityBins {
    static constexpr int64_t count = 24;  // a quarter-decade each
    static constexpr int64_t log10_lo = -4;
    static constexpr int64_t log10_hi = 2;
};

// Log10 bins of a page's age in TOKENS, stream_len minus the page's base address. The
// range must cover a whole run's stream length, since a page that is never evicted ages
// with the stream. EngineBinding::stats publishes these edges and consumers label their
// axes from them, so they are declared once here.
struct AgeBins {
    static constexpr int64_t count = 24;  // a quarter-decade each
    static constexpr int64_t log10_lo = 0;
    static constexpr int64_t log10_hi = 6;
};

// Linear bins of a rerank score squashed to r / (1 + r), so parity with the bar sits
// at 0.5 and nothing clips.
struct RecallBins {
    static constexpr int64_t count = 20;
    static constexpr int64_t lo = 0;
    static constexpr int64_t hi = 1;
};

// Per-seq evict->recall accounting over a run. Counts PAGES actually demoted and
// recalled, not the requested budget. Reset to {} when a sequence's per-run state
// resets (create / continuation).
//
// staged_hits/staged_misses count PREFILTERED CANDIDATES, not cycles: a candidate
// already resident in the staging cache against one read out of a bucket.
// promoted_s1/promoted_s2 split recalled_pages by originating bucket, so
// promoted_s2 / recalled_pages is the disk share of PROMOTIONS, not of candidates.
//
// The histograms bin every candidate as scored, before selection, so they
// are independent of the bar. recall_hist covers both phases; recall_hist_decode is the
// decode-cadence cycles alone, which score against the sequence's own generated tokens
// rather than against prompt text.
//
// Every _decode field is a SUBSET of the plainly named one, which covers both phases;
// the prefill half is the difference. That holds for the eviction attribution too:
//
// bar_evicted_pages + backstop_evicted_pages == evicted_pages. The first cleared the
// keep-bar test, the second were taken afterwards to hold active_max_size.
//
// evicted_age_hist bins each demoted page's age in tokens (see AgeBins) and
// candidate_age_hist every evictable page's, victims included, so the two compare.
//
// raw_mass_agree_pages of raw_mass_victim_pages is the overlap between the real victim
// set and the one an identically sized selection over the UNCORRECTED mass would take,
// so the ratio is how much the age correction moves the decision.
struct Stats {
    int64_t evicted_pages = 0;
    int64_t bar_evicted_pages = 0;
    int64_t bar_evicted_pages_decode = 0;
    int64_t backstop_evicted_pages = 0;
    int64_t backstop_evicted_pages_decode = 0;
    int64_t raw_mass_victim_pages = 0;
    int64_t raw_mass_victim_pages_decode = 0;
    int64_t raw_mass_agree_pages = 0;
    int64_t raw_mass_agree_pages_decode = 0;
    int64_t recalled_pages = 0;
    int64_t cycles = 0;
    // Residency swings within a cycle: eviction takes it to the trough, recall brings
    // it back. Active length in TOKENS summed at both points, so cycles divides either
    // into a mean and the difference is what recall restored.
    int64_t post_evict_tokens = 0;
    int64_t post_evict_tokens_decode = 0;
    int64_t post_recall_tokens = 0;
    int64_t post_recall_tokens_decode = 0;
    // Cycles whose eviction left no room: recall returns before scoring anything.
    int64_t starved_cycles = 0;
    int64_t starved_cycles_decode = 0;
    int64_t decode_cycles = 0;
    int64_t staged_hits = 0;
    int64_t staged_misses = 0;
    int64_t promoted_s1 = 0;
    int64_t promoted_s2 = 0;
    std::array<int64_t, DensityBins::count> density_hist{};
    std::array<int64_t, AgeBins::count> evicted_age_hist{};
    std::array<int64_t, AgeBins::count> evicted_age_hist_decode{};
    std::array<int64_t, AgeBins::count> candidate_age_hist{};
    std::array<int64_t, AgeBins::count> candidate_age_hist_decode{};
    std::array<int64_t, RecallBins::count> recall_hist{};
    std::array<int64_t, RecallBins::count> recall_hist_decode{};
    DensityByPosition density_by_position{};
};

// Underflow (v <= 0 included) lands in bin 0, overflow in the last bin.
inline int64_t density_bin(double v) {
    if (!(v > 0.0)) {
        return 0;
    }
    constexpr double lo = DensityBins::log10_lo;
    constexpr double decades = DensityBins::log10_hi - DensityBins::log10_lo;
    const int64_t b = static_cast<int64_t>(std::floor((std::log10(v) - lo) / decades * DensityBins::count));
    return std::clamp<int64_t>(b, 0, DensityBins::count - 1);
}

inline int64_t age_bin(double v) {
    if (!(v > 0.0)) {
        return 0;
    }
    constexpr double lo = AgeBins::log10_lo;
    constexpr double decades = AgeBins::log10_hi - AgeBins::log10_lo;
    const int64_t b = static_cast<int64_t>(std::floor((std::log10(v) - lo) / decades * AgeBins::count));
    return std::clamp<int64_t>(b, 0, AgeBins::count - 1);
}

inline int64_t recall_bin(double s) {
    if (!(s > 0.0)) {
        return 0;
    }
    constexpr double span = RecallBins::hi - RecallBins::lo;
    const int64_t b = static_cast<int64_t>(std::floor((s - RecallBins::lo) / span * RecallBins::count));
    return std::clamp<int64_t>(b, 0, RecallBins::count - 1);
}

struct Engine {
    // Builds and owns Model + ActiveBuffer + Scheduler + Sampler + the recall
    // components. Build order: ActiveBuffer (pool) -> Model -> Scheduler -> Sampler
    // -> RecallBuffer + s1/s2 + KVMemory (from the pool geometry, when recall is on).
    explicit Engine(EngineConfig config);

    // Open a new session and return its engine-assigned seq_id. The non-empty system
    // prompt sets the protected sink (n_sink = its length) and prefills as the
    // session's first tokens; content turns follow via feed(). create_session() just
    // calls feed() on a fresh seq.
    //
    // `overrides` is this session's SessionConfig, already merged over the engine's
    // defaults by the caller; it is resolved and frozen here, so a malformed value fails
    // now rather than mid-run and a run stays reproducible from its inputs. A failed
    // create_session() burns the id it drew (never reused) but opens nothing.
    int64_t create_session(c10::List<int64_t> system_tokens, const SessionConfig& overrides);
    int64_t create_session(c10::List<int64_t> system_tokens) {
        return this->create_session(std::move(system_tokens), this->default_session);
    }
    // The engine's session defaults, for a caller merging a partial override dict.
    const SessionConfig& session_defaults() const {
        return this->default_session;
    }
    // Enqueue tokens onto a sequence; no model forward here, the next step() runs
    // the batched prefill. On a fresh seq (from create) it starts the session, with
    // n_sink = this feed's length. On a seq opened but not yet prefilled it extends
    // the initial prefill (n_sink unchanged). On a prior turn with active KV it
    // continues, prefilling addresses [pending_base, stream_len) over the active KV;
    // pending_base is one token early so the prior turn's unwritten lag token is
    // re-prefilled. A still-generating sequence is parked first, so the caller ends a
    // turn simply by feeding the next one.
    //
    // Empty tokens are a noop on an unknown seq_id, and still (re)schedule a known one.
    void feed(int64_t seq_id, c10::List<int64_t> tokens);

    // Unbounded (nullopt) until step()ing hits eos_id.
    void set_decode_budget(int64_t seq_id, std::optional<int64_t> budget);

    // Emit up to n tokens, returned in emission order (prefill group then decode
    // group, per scheduler iteration). Iterations run until n tokens are emitted or
    // no sequence is active; an iteration that emits nothing (a prompt chunk that
    // does not complete a prefill) does not count against n, so the first step()
    // after a feed() drives the whole prefill and takes far longer than later ones.
    // With several active sequences one iteration emits one token per sequence and
    // is not divisible, so the result can overshoot n by up to (active sequences - 1).
    // Returns short of n when every sequence finishes.
    std::vector<TokenEvent> step(int64_t n);

    // Sequences still running or waiting in the scheduler.
    int64_t active_sequence_count() const;

    // Free a sequence's active KV (if any), its demoted KV, and host state.
    void release(int64_t seq_id);

    // A sequence's scheduling state right now: active (queued or running in the
    // scheduler), parked (KV retained, between a finished turn and the next
    // feed()), unknown (never created, or already released).
    enum class SessionState { active, parked, unknown };
    SessionState session_state(int64_t seq_id) const;

    // A sequence's demoted tokens across the bucket chain (0 when recall off). The
    // RecallBuffer holds transient copies and is not counted.
    int64_t demoted_tokens_count(int64_t seq_id) const;

    // Attended KV tokens for a sequence (its active length), 0 if it has none.
    int64_t active_tokens_count(int64_t seq_id) const;

    // Per-seq evict->recall accounting (pages demoted/recalled, cycles) since the
    // last per-run reset. Unknown seq is a TORCH_CHECK.
    Stats stats(int64_t seq_id) const;

    // Attention context: the ACTIVE pages, in ascending address (== page position ==
    // attention/RoPE) order -- the window the attention kernels see. Works with recall
    // on or off. page_ids gives the page indices, pages gives each page's token ids.
    std::vector<int64_t> context_page_ids(int64_t seq_id) const;
    std::vector<std::vector<int64_t>> context_pages(int64_t seq_id) const;

    // Full conversation history: the pages merged across the active buffer and the
    // bucket chain in address order. Requires recall (the KVMemory). A
    // just-generated token whose KV is not yet written (the trailing lag) is absent.
    std::vector<int64_t> history_page_ids(int64_t seq_id) const;
    std::vector<std::vector<int64_t>> history_pages(int64_t seq_id) const;

  private:
    struct SeqState {
        // The full token stream is NOT stored: prefilled/generated ids live in the KV
        // memory (recoverable via history() until release), the most recent sampled
        // token is last_token, and this turn's un-prefilled tokens sit in a small
        // pending buffer at addresses [pending_base, pending_base + pending.size())
        // that clears once the prefill lands.
        std::vector<int64_t> pending;
        int64_t pending_base = 0;
        int64_t stream_len = 0;  // total conversation length (addresses [0, stream_len))
        int64_t prompt_len = 0;  // this turn's prompt end; the prefill boundary
        int64_t last_token = -1;  // most recent sampled token id (decode input)
        // Tokens this turn may still generate; -1 is unbounded. Sampling decrements it
        // and Engine::finished treats 0 as a finish, same as eos_id, so a turn's cap is
        // enforced inside the same step() that samples -- checking it in Python after
        // step() returns cannot stop a turn already mid-step (see feed()).
        int64_t budget = -1;
        GenerationParams params;
        // This session's resolved knobs, fixed at create(). Every evict->recall decision
        // reads them from here, never from the Engine, so two live sessions may differ.
        SessionRuntime cfg;
        at::Generator gen;  // per-seq RNG, seeded from params.seed at first feed
        // Protected sink prefix [0, n_sink) never evicted; set on the fresh prefill.
        int64_t n_sink = 0;
        // The session's short_offset with the 0 sentinel resolved to n_sink, set by the
        // same fresh prefill that sets the sink. Every consumer of the offset reads
        // THIS, never SessionRuntime's configured value.
        int64_t short_offset = 0;
        // Decode-cycle cadence: tokens generated since the last evict->recall
        // cycle; reset on feed and every time the cycle fires.
        int64_t decode_since_cycle = 0;
        // Tokens this sequence has forwarded, the clock the spill's lazy mass decay
        // measures age in. The decay rate is per TOKEN, so a per-step clock would
        // under-decay a prefill step by its whole chunk. Counts the same tokens
        // ActiveBuffer::upkeep is called with.
        int64_t forwarded_tokens = 0;
        // Per-seq evict->recall accounting; reset with the per-run state.
        Stats stats;
    };

    // Token ids for addresses [start, end): the pending buffer for this turn's
    // un-prefilled tail, the KV memory's segment documents below it.
    std::vector<int64_t> token_ids_at(int64_t seq_id, int64_t start, int64_t end);

    using AcceptPlan = std::vector<std::pair<int64_t, int64_t>>;  // (seq_id, row)

    // One scheduler step: build groups, one forward, sample, update/park; returns
    // the tokens emitted this step. step(n) loops this.
    std::vector<TokenEvent> step_once();

    // Build the prefill/decode GroupBatch and its accept plan; nullopt when the
    // group is empty.
    std::optional<GroupBatch> build_prefill(const BatchDescriptor& desc, AcceptPlan& accept);
    std::optional<GroupBatch> build_decode(const BatchDescriptor& desc, AcceptPlan& accept);
    void accept_group(
        const at::Tensor& logits,
        const AcceptPlan& accept,
        std::vector<int64_t>& ids,
        std::vector<int64_t>& tokens,
        std::vector<bool>& finished
    );
    bool finished(const SeqState& st, int64_t token) const;

    // Build the RecallBuffer, the s1/s2 buckets and the KVMemory from the pool
    // geometry when recall is on. spill_dir empty => no s2.
    void configure_recall(const std::string& spill_dir);

    // Apply the defaults, convert each token-valued knob into the unit its site works
    // in, turn both taus into density bars, and parse the head-reduce string. Throws on
    // a malformed value, so create() is where a bad session config fails.
    SessionRuntime resolve_session(const SessionConfig& cfg) const;

    // Start a session on a seq the engine does not know yet: seed its resolved knobs,
    // make this feed the protected sink, and queue the prefill. create() passes the
    // session's own resolved config; a feed() onto an unknown seq passes the engine
    // defaults'.
    void open_session(int64_t seq_id, c10::List<int64_t> tokens, const SessionRuntime& cfg);

    // What the rerank scores candidates against. token_ids feed the BM25 prefilter;
    // vecs are the query tokens' pre-RoPE Q and lse the softmax denominator each of
    // those tokens attended under, one per relevance layer; log_attended_len is [T] fp32,
    // the log of the keys each query row attended. Q and LSE must be gathered BEFORE
    // eviction shrinks the pool.
    struct RecallQuery {
        std::vector<int64_t> token_ids;
        std::vector<at::Tensor> vecs;  // per layer [T, n_heads, head_dim]
        std::vector<at::Tensor> lse;  // per layer [T, n_heads]
        at::Tensor log_attended_len;
    };

    // Engine-owned evict->recall cycle, once per step after update. A sequence
    // prefilling this step cycles per chunk with that chunk as the query; the rest cycle
    // once per decode_cycle_interval generated tokens with their recent output as the
    // query.
    void recall_cycle(const BatchDescriptor& desc);
    // One sequence's cycle: evict to the bar, then refill the freed headroom.
    void run_cycle(int64_t seq_id, const RecallQuery& query, bool decode);
    // Demote every evictable page whose density is under the bar, with active_max_size
    // as a backstop, then hand the victims to the KV memory. No-op with no evictable
    // band. decode is the calling cycle's cadence and only splits the counters.
    void evict_demote(int64_t seq_id, bool decode);
    // The buffer positions of a cycle's query rows, int64 [T] on the host: row i is the
    // slot at active_len - T + i. Must be called AFTER the cycle's eviction.
    at::Tensor query_row_positions(int64_t seq_id, const RecallQuery& query) const;
    // The position the rerank scores a candidate at: short_offset under the compacted
    // layout, the slot below the protected working run under contiguous. Must be called
    // AFTER the cycle's eviction.
    int64_t candidate_scoring_position(int64_t seq_id) const;
    // Headroom-bounded recall: capacity is active_max_size - active_len, the whole
    // headroom eviction freed, so active_len stays under active_max_size. Selection
    // stops itself on that capacity; recall_tau and recall_temperature decide which
    // candidates take it. Promoted pages are merged with survivors in address order and
    // the page table permuted to match, re-roping shifted keys in place. No-op when
    // there is no capacity or nothing is recalled.
    void recall_recompact(int64_t seq_id, const RecallQuery& query, bool decode);
    // End of one sequence's evict->recall cycle, after promote: release the buckets'
    // deferred slot frees and run the s1 -> s2 spill to ram_bucket_size.
    void end_cycle(int64_t seq_id);
    // The PREFILL group's captured pre-RoPE Q and softmax denominator for rows
    // [row_lo, row_hi), one tensor per relevance layer, left on the model device where
    // the rerank scores. A decode cycle reads its query from the Q ring instead.
    std::vector<at::Tensor> gather_query_q(int64_t row_lo, int64_t row_hi) const;
    std::vector<at::Tensor> gather_query_lse(int64_t row_lo, int64_t row_hi) const;

    // The last recall_query_window decoded tokens' Q per relevance layer, on the model
    // device. A decode cycle scores with query tokens from earlier steps, so it needs the
    // denominator and attended key count of the step that produced each one, not of the
    // step that runs the cycle. Only allocated when recall is on.
    struct QRing {
        std::vector<at::Tensor> buf;  // per layer [window, n_heads, head_dim]
        std::vector<at::Tensor> lse;  // per layer [window, n_heads]
        std::vector<int64_t> ctx;  // [window] attended key count per slot
        int64_t cursor = 0;  // next write slot
        int64_t count = 0;  // valid tokens, capped at window

        // The valid slots oldest first: buf[0, count) before the first wrap, starting at
        // cursor once full.
        std::vector<int64_t> chronological(int64_t window) const;
    };
    // After a decode forward, append each decode-group seq's just-decoded token Q to
    // its ring (allocating lazily). No-op for the prefill group and when recall is off.
    void update_q_ring(const BatchDescriptor& desc);
    // The seq's ring as a recall query: Q, LSE and log attended length over the last
    // min(window, count) tokens in chronological order. token_ids is left empty for the
    // caller to fill. Empty vecs when the seq has no ring or no buffered Q.
    RecallQuery read_q_ring(int64_t seq_id) const;

    std::unique_ptr<ActiveBuffer> active;
    std::unique_ptr<Model> model;
    std::unique_ptr<Scheduler> sched;
    std::unique_ptr<Sampler> sampler;
    GenerationParams default_params;
    int64_t eos_id = -1;
    at::Device device;
    std::unordered_map<int64_t, SeqState> seqs;
    int64_t next_seq_id = 0;  // engine-assigned; see create_session

    int64_t active_buffer_size;
    int64_t max_chunk_size;
    int64_t max_sessions;  // sessions that may hold pool capacity at once
    int64_t page_size;  // the pool's, for resolving a session's token-valued knobs
    // The token count the evict->recall cycle works to: eviction demotes toward it,
    // recall fills only the headroom below it, and it seeds the attention-mass EMA rate.
    // max(0, active_buffer_size - max_chunk_size), so it is the same for every session.
    // 0 when the buffer is smaller than one chunk, which is legal only with recall off.
    int64_t active_max_size;
    // The length the density is stated against, in TOKENS; <= 0 uses each query's own
    // attended key count (see EngineConfig).
    int64_t mass_reference_length;
    // What the attention kernels multiply each query's mass by. A configured reference
    // length is a CONSTANT and commutes with the accumulation, so the kernels take 1
    // and ActiveBuffer restores it once per page; only the per-query attended count
    // has to travel into the kernels, and 0 is what asks for it.
    double mass_length_gain() const {
        return this->mass_reference_length > 0 ? 1.0 : 0.0;
    }
    // Generated tokens between a sequence's decode-cadence cycles, resolved: always > 0.
    int64_t decode_cycle_interval;
    // What a session's knobs default to when create() names no override.
    SessionConfig default_session;
    RecallConfig recall_cfg;
    // Null when recall is off (recall_buffer_size <= 0). s1 holds the demoted pages on
    // host RAM; s2 is the disk overflow behind it, disabled when spill_dir is empty.
    // recall_buffer is device staging for one cycle's candidate set, not a tier.
    std::unique_ptr<RecallBuffer> recall_buffer;
    std::unique_ptr<PagedBucket> s1;
    std::unique_ptr<FileBucket> s2;
    std::unique_ptr<KVMemory> memory;
    int64_t recall_buffer_size;
    int64_t ram_bucket_size;
    bool recall_skip_fresh;
    // Per-seq decode-token Q ring buffers (the rerank's multi-token query).
    std::unordered_map<int64_t, QRing> q_rings;
};

}  // namespace pulsar
