#include "pulsar/runtime/engine.hpp"

#include <torch/custom_class.h>
#include <torch/library.h>

#include <ATen/core/Dict.h>
#include <ATen/core/List.h>
#include <c10/util/intrusive_ptr.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// Thin TorchBind shim over the plain-C++ Engine (registered as
// torch.classes.pulsar.Engine), the only Python-facing serving class. Owns a
// unique_ptr<Engine> and forwards the Python-facing surface.
//
// __init__ takes three args: the native weight dict (empty {} for the compiled path),
// a config dict keyed by EngineConfig field name, and dtype (see EngineBinding above).
// Config values are typed per field: model, device, decode_pt2, prefill_pt2, spill_dir
// and active_position_layout are strings; rope_theta, rms_eps, temperature, top_p,
// recall_tau, recall_temperature, evict_tau and attention_mass_decay are doubles;
// recall_skip_fresh is a bool; relevance_layers an int list; the rest int64. A key the
// engine does not know is REJECTED, so a stale key cannot silently do nothing; an
// absent key keeps EngineConfig's default. model (a family+precision id) selects the
// construction path. decode_cycle_interval <= 0 takes max_chunk_size;
// mass_reference_length <= 0 states the eviction/recall density against each query's own
// attended key count; every other key is used as given.
//
// The per-session keys (n_working, recall_query_window, block_radius,
// active_position_layout, short_offset, recall_segment_size, recall_prefilter_k,
// recall_bm25_query_tokens, recall_tau, evict_tau, recall_temperature,
// attention_mass_decay)
// are FLAT in that dict and seed the engine's session DEFAULTS. create() takes the same
// key names to override them for one session; everything else is engine-level and is
// rejected there.

namespace pulsar {

struct EngineBinding : torch::CustomClassHolder {
    // dtype is its own argument, not a "dtype" config key: a torch.dtype value boxed
    // into config's generic Dict[str, Any] fails pybind's per-key IValue conversion.
    EngineBinding(
        c10::Dict<std::string, at::Tensor> weights,
        c10::Dict<std::string, c10::IValue> config,
        at::ScalarType dtype
    );

    // The non-empty system prompt sets the protected sink (n_sink) and prefills as
    // the session's first tokens; content turns follow via feed(). overrides is this
    // session's per-session knobs, by the same flat key names the engine config dict
    // uses. An absent key inherits the engine's default; an unknown or engine-level
    // key is rejected. The values are resolved and frozen here, so a malformed one
    // fails at create_session() rather than mid-run.
    int64_t create_session(c10::List<int64_t> system_tokens, c10::Dict<std::string, c10::IValue> overrides);
    // Extends the initial prefill when it has not started yet, or starts a
    // continuation prefill onto a prior turn's KV, parking the sequence first when
    // it is still generating. Empty tokens on an unknown seq_id is a noop; on a
    // known one it still (re)schedules the seq.
    void feed(int64_t seq_id, c10::List<int64_t> tokens);
    // This turn's token cap; absent (None) leaves it unbounded, running until eos_id.
    void set_decode_budget(int64_t seq_id, std::optional<int64_t> budget);
    // (seq_ids, tokens, finished) for up to n emitted tokens, as parallel lists. Any
    // pending prefill runs to completion inside the call; see Engine::step.
    std::tuple<c10::List<int64_t>, c10::List<int64_t>, c10::List<bool>> step(int64_t n);
    void release(int64_t seq_id);
    // One of "active", "parked" or "unknown".
    std::string session_state(int64_t seq_id) const;
    int64_t demoted_tokens_count(int64_t seq_id) const;
    int64_t active_tokens_count(int64_t seq_id) const;
    // Per-seq evict->recall accounting since the last per-run reset. A field marked
    // [decode] has a "_decode" twin holding the decode-cadence share alone (the plain
    // key covers both phases; prefill's share is the difference). cycles is the one
    // exception: its decode-cadence twin is named decode_cycles, not cycles_decode.
    // Fields with no [decode] mark have no twin at all.
    //
    // Int64 scalars:
    //   evicted_pages: pages demoted this run.
    //   bar_evicted_pages [decode]: ...that cleared the keep-bar test.
    //   backstop_evicted_pages [decode]: ...taken after, to hold active_max_size.
    //       bar_evicted_pages + backstop_evicted_pages == evicted_pages.
    //   raw_mass_victim_pages [decode]: pages an identically sized selection over the
    //       UNCORRECTED mass would have evicted (the age-correction counterfactual).
    //   raw_mass_agree_pages [decode]: ...of those, how many overlap the real
    //       (corrected) victim set; the ratio is how much age correction moves the
    //       decision.
    //   recalled_pages: pages recalled.
    //   cycles [decode: decode_cycles]: evict->recall cycles run.
    //   post_evict_tokens [decode]: active tokens summed right after eviction, the
    //       residency trough.
    //   post_recall_tokens [decode]: active tokens summed right after recall, the
    //       residency peak. cycles divides either into a mean.
    //   starved_cycles [decode]: cycles whose eviction left no room, so recall
    //       returned before scoring anything.
    //   staged_hits: prefiltered candidates already resident in the staging cache.
    //   staged_misses: ...vs read out of a bucket.
    //   promoted_s1: recalled pages that came from the S1 bucket.
    //   promoted_s2: ...from the S2 bucket.
    //
    // Int64-list histograms, each self-describing via its own edge scalars:
    //   density_hist: density_log10_lo, density_log10_hi, density_nbins.
    //   evicted_age_hist [decode]: age_log10_lo, age_log10_hi, age_nbins.
    //   candidate_age_hist [decode]: age_log10_lo, age_log10_hi, age_nbins.
    //   recall_hist [decode]: recall_lo, recall_hi, recall_nbins.
    // evicted_age_hist bins each demoted page's age in tokens, candidate_age_hist every
    // evictable page's (victims included), so the two compare. Consumers label their
    // axes from the edge scalars rather than restating them.
    //
    // Density-by-position probe, over "position_nbuckets" normalized-position buckets:
    //   position_raw_mass_sum: double list; per-bucket SUM, not mean.
    //   position_corrected_density_sum: double list; per-bucket SUM, not mean.
    //   position_page_count: int64 list; pages behind each bucket (the consumer
    //       divides for a mean).
    c10::Dict<std::string, c10::IValue> stats(int64_t seq_id) const;
    // The ACTIVE pages in attention/RoPE order (not merged with the demoted ones),
    // as page indices (page_ids) and per-page token ids (pages). Works with recall
    // on or off.
    c10::List<int64_t> context_page_ids(int64_t seq_id) const;
    c10::List<c10::List<int64_t>> context_pages(int64_t seq_id) const;
    // The active, demoted and spilled pages merged in address order. Requires
    // recall.
    c10::List<int64_t> history_page_ids(int64_t seq_id) const;
    c10::List<c10::List<int64_t>> history_pages(int64_t seq_id) const;

  private:
    std::unique_ptr<Engine> engine;
};

namespace {

// Any contiguous range of ints/doubles -> the c10::List the TorchBind surface returns.
template <typename R> c10::List<int64_t> to_int_list(const R& values) {
    c10::List<int64_t> out;
    out.reserve(std::size(values));
    for (int64_t v : values) {
        out.push_back(v);
    }
    return out;
}

template <typename R> c10::List<double> to_double_list(const R& values) {
    c10::List<double> out;
    out.reserve(std::size(values));
    for (double v : values) {
        out.push_back(v);
    }
    return out;
}

c10::List<c10::List<int64_t>> to_page_list(const std::vector<std::vector<int64_t>>& pages) {
    c10::List<c10::List<int64_t>> out;
    out.reserve(pages.size());
    for (const std::vector<int64_t>& page : pages) {
        out.push_back(to_int_list(page));
    }
    return out;
}

// Reads a config dict and records which keys it consumed, so a key it does not know is
// rejected instead of silently ignored. A field keeps the value it came in with when its
// key is absent. context names the dict in the rejection message.
struct ConfigReader {
    const c10::Dict<std::string, c10::IValue>& config;
    std::string context;
    std::set<std::string> consumed;

    bool take(const std::string& key) {
        this->consumed.insert(key);
        return this->config.contains(key);
    }
    void operator()(const std::string& key, int64_t& field) {
        if (this->take(key)) {
            field = this->config.at(key).toInt();
        }
    }
    void operator()(const std::string& key, double& field) {
        if (this->take(key)) {
            field = this->config.at(key).toDouble();
        }
    }
    void operator()(const std::string& key, std::string& field) {
        if (this->take(key)) {
            field = this->config.at(key).toStringRef();
        }
    }
    void operator()(const std::string& key, bool& field) {
        if (this->take(key)) {
            field = this->config.at(key).toBool();
        }
    }
    void operator()(const std::string& key, std::vector<int64_t>& field) {
        if (!this->take(key)) {
            return;
        }
        field.clear();
        for (int64_t v : this->config.at(key).toIntVector()) {
            field.push_back(v);
        }
    }

    void reject_unknown() const {
        std::vector<std::string> unknown;
        for (const auto& entry : this->config) {
            if (!this->consumed.count(entry.key())) {
                unknown.push_back(entry.key());
            }
        }
        if (unknown.empty()) {
            return;
        }
        std::sort(unknown.begin(), unknown.end());
        std::string names;
        for (const std::string& k : unknown) {
            names += (names.empty() ? "" : ", ") + k;
        }
        TORCH_CHECK(false, this->context, ": unknown key(s): ", names);
    }
};

// The per-session knobs, keyed flat. Read from the engine config dict as the session
// DEFAULTS and from create()'s dict as one session's overrides, so both surfaces name a
// knob identically and neither can drift from the other.
void read_session(ConfigReader& set, SessionConfig& session) {
    set("n_working", session.n_working);
    set("recall_query_window", session.recall_query_window);
    set("block_radius", session.block_radius);
    set("active_position_layout", session.active_position_layout);
    set("short_offset", session.short_offset);
    set("recall_segment_size", session.recall_segment_size);
    set("recall_prefilter_k", session.recall_prefilter_k);
    set("recall_bm25_query_tokens", session.recall_bm25_query_tokens);
    set("recall_tau", session.recall_tau);
    set("evict_tau", session.evict_tau);
    set("recall_temperature", session.recall_temperature);
    set("attention_mass_decay", session.attention_mass_decay);
    set("recall_during_prefill", session.recall_during_prefill);
}

}  // namespace

EngineBinding::EngineBinding(
    c10::Dict<std::string, at::Tensor> weights,
    c10::Dict<std::string, c10::IValue> config,
    at::ScalarType dtype
) {
    EngineConfig cfg;
    cfg.weights = std::move(weights);
    cfg.dtype = dtype;
    ConfigReader set{config, "Engine config"};
    set("model", cfg.model);
    set("decode_pt2", cfg.decode_pt2);
    set("prefill_pt2", cfg.prefill_pt2);
    set("max_sessions", cfg.max_sessions);
    set("n_layers", cfg.n_layers);
    set("n_heads", cfg.n_heads);
    set("n_kv_heads", cfg.n_kv_heads);
    set("head_dim", cfg.head_dim);
    set("hidden", cfg.hidden);
    set("intermediate", cfg.intermediate);
    set("vocab", cfg.vocab);
    set("rope_theta", cfg.rope_theta);
    set("rms_eps", cfg.rms_eps);
    set("active_buffer_size", cfg.active_buffer_size);
    set("page_size", cfg.page_size);
    set("device", cfg.device);
    set("max_running", cfg.max_running);
    set("max_chunk_size", cfg.max_chunk_size);
    set("mass_reference_length", cfg.mass_reference_length);
    set("eos_id", cfg.eos_id);
    set("temperature", cfg.temperature);
    set("top_p", cfg.top_p);
    set("top_k", cfg.top_k);
    set("seed", cfg.seed);
    read_session(set, cfg.session);
    set("relevance_layers", cfg.relevance_layers);
    set("recall_buffer_size", cfg.recall_buffer_size);
    set("ram_bucket_size", cfg.ram_bucket_size);
    set("spill_dir", cfg.spill_dir);
    set("recall_skip_fresh", cfg.recall_skip_fresh);
    set("decode_cycle_interval", cfg.decode_cycle_interval);
    set.reject_unknown();
    this->engine = std::make_unique<Engine>(std::move(cfg));
}

int64_t EngineBinding::create_session(c10::List<int64_t> system_tokens, c10::Dict<std::string, c10::IValue> overrides) {
    // Merge over the engine's defaults, so an absent key inherits rather than resetting
    // to SessionConfig's own default.
    SessionConfig session = this->engine->session_defaults();
    ConfigReader set{overrides, "create_session: session override"};
    read_session(set, session);
    set.reject_unknown();
    return this->engine->create_session(std::move(system_tokens), session);
}

void EngineBinding::feed(int64_t seq_id, c10::List<int64_t> tokens) {
    this->engine->feed(seq_id, std::move(tokens));
}

void EngineBinding::set_decode_budget(int64_t seq_id, std::optional<int64_t> budget) {
    this->engine->set_decode_budget(seq_id, budget);
}

std::tuple<c10::List<int64_t>, c10::List<int64_t>, c10::List<bool>> EngineBinding::step(int64_t n) {
    std::vector<TokenEvent> events = this->engine->step(n);
    c10::List<int64_t> seq_ids, tokens;
    c10::List<bool> finished;
    for (const TokenEvent& e : events) {
        seq_ids.push_back(e.seq_id);
        tokens.push_back(e.token);
        finished.push_back(e.finished);
    }
    return std::make_tuple(std::move(seq_ids), std::move(tokens), std::move(finished));
}

void EngineBinding::release(int64_t seq_id) {
    this->engine->release(seq_id);
}

std::string EngineBinding::session_state(int64_t seq_id) const {
    switch (this->engine->session_state(seq_id)) {
    case Engine::SessionState::active:
        return "active";
    case Engine::SessionState::parked:
        return "parked";
    default:
        return "unknown";
    }
}

int64_t EngineBinding::demoted_tokens_count(int64_t seq_id) const {
    return this->engine->demoted_tokens_count(seq_id);
}

int64_t EngineBinding::active_tokens_count(int64_t seq_id) const {
    return this->engine->active_tokens_count(seq_id);
}

c10::Dict<std::string, c10::IValue> EngineBinding::stats(int64_t seq_id) const {
    Stats s = this->engine->stats(seq_id);
    // Heterogeneous value type: scalar counters/edges (int64), int64 histograms and
    // double per-position sums share one dict, so the value type is Any.
    // Dict<string, IValue> has no public constructor, so build a GenericDict and
    // convert.
    c10::impl::GenericDict out(c10::StringType::get(), c10::AnyType::get());
    out.insert(std::string("evicted_pages"), s.evicted_pages);
    out.insert(std::string("bar_evicted_pages"), s.bar_evicted_pages);
    out.insert(std::string("bar_evicted_pages_decode"), s.bar_evicted_pages_decode);
    out.insert(std::string("backstop_evicted_pages"), s.backstop_evicted_pages);
    out.insert(std::string("backstop_evicted_pages_decode"), s.backstop_evicted_pages_decode);
    out.insert(std::string("raw_mass_victim_pages"), s.raw_mass_victim_pages);
    out.insert(std::string("raw_mass_victim_pages_decode"), s.raw_mass_victim_pages_decode);
    out.insert(std::string("raw_mass_agree_pages"), s.raw_mass_agree_pages);
    out.insert(std::string("raw_mass_agree_pages_decode"), s.raw_mass_agree_pages_decode);
    out.insert(std::string("recalled_pages"), s.recalled_pages);
    out.insert(std::string("cycles"), s.cycles);
    out.insert(std::string("decode_cycles"), s.decode_cycles);
    out.insert(std::string("post_evict_tokens"), s.post_evict_tokens);
    out.insert(std::string("post_evict_tokens_decode"), s.post_evict_tokens_decode);
    out.insert(std::string("post_recall_tokens"), s.post_recall_tokens);
    out.insert(std::string("post_recall_tokens_decode"), s.post_recall_tokens_decode);
    out.insert(std::string("starved_cycles"), s.starved_cycles);
    out.insert(std::string("starved_cycles_decode"), s.starved_cycles_decode);
    out.insert(std::string("staged_hits"), s.staged_hits);
    out.insert(std::string("staged_misses"), s.staged_misses);
    out.insert(std::string("promoted_s1"), s.promoted_s1);
    out.insert(std::string("promoted_s2"), s.promoted_s2);
    out.insert(std::string("density_hist"), to_int_list(s.density_hist));
    out.insert(std::string("density_log10_lo"), DensityBins::log10_lo);
    out.insert(std::string("density_log10_hi"), DensityBins::log10_hi);
    out.insert(std::string("density_nbins"), DensityBins::count);
    out.insert(std::string("evicted_age_hist"), to_int_list(s.evicted_age_hist));
    out.insert(std::string("evicted_age_hist_decode"), to_int_list(s.evicted_age_hist_decode));
    out.insert(std::string("candidate_age_hist"), to_int_list(s.candidate_age_hist));
    out.insert(std::string("candidate_age_hist_decode"), to_int_list(s.candidate_age_hist_decode));
    out.insert(std::string("age_log10_lo"), AgeBins::log10_lo);
    out.insert(std::string("age_log10_hi"), AgeBins::log10_hi);
    out.insert(std::string("age_nbins"), AgeBins::count);
    out.insert(std::string("recall_hist"), to_int_list(s.recall_hist));
    out.insert(std::string("recall_hist_decode"), to_int_list(s.recall_hist_decode));
    out.insert(std::string("recall_lo"), RecallBins::lo);
    out.insert(std::string("recall_hi"), RecallBins::hi);
    out.insert(std::string("recall_nbins"), RecallBins::count);
    const DensityByPosition& by_pos = s.density_by_position;
    out.insert(std::string("position_raw_mass_sum"), to_double_list(by_pos.raw_mass_sum));
    out.insert(std::string("position_corrected_density_sum"), to_double_list(by_pos.corrected_density_sum));
    out.insert(std::string("position_page_count"), to_int_list(by_pos.page_count));
    out.insert(std::string("position_nbuckets"), DensityByPosition::buckets);
    return c10::impl::toTypedDict<std::string, c10::IValue>(std::move(out));
}

c10::List<int64_t> EngineBinding::context_page_ids(int64_t seq_id) const {
    return to_int_list(this->engine->context_page_ids(seq_id));
}

c10::List<c10::List<int64_t>> EngineBinding::context_pages(int64_t seq_id) const {
    return to_page_list(this->engine->context_pages(seq_id));
}

c10::List<int64_t> EngineBinding::history_page_ids(int64_t seq_id) const {
    return to_int_list(this->engine->history_page_ids(seq_id));
}

c10::List<c10::List<int64_t>> EngineBinding::history_pages(int64_t seq_id) const {
    return to_page_list(this->engine->history_pages(seq_id));
}

}  // namespace pulsar

// TORCH_LIBRARY_FRAGMENT adds to the pulsar namespace defined in ops.cpp.
// EngineBinding is the only class registered for Python (torch.classes.pulsar.Engine);
// the engine internals are plain C++ and unregistered. See the EngineBinding struct
// above for the __init__ (weights, config, dtype) form.
TORCH_LIBRARY_FRAGMENT(pulsar, m) {
    m.class_<pulsar::EngineBinding>("Engine")
        .def(torch::init<c10::Dict<std::string, at::Tensor>, c10::Dict<std::string, c10::IValue>, at::ScalarType>())
        // overrides defaults to an empty dict: a caller that names no per-session knob
        // inherits every engine default. Returns the engine-assigned seq_id.
        .def(
            "create_session",
            &pulsar::EngineBinding::create_session,
            "",
            {torch::arg("system_tokens"),
             torch::arg("overrides") = c10::IValue(c10::impl::GenericDict(c10::StringType::get(), c10::AnyType::get()))}
        )
        .def("feed", &pulsar::EngineBinding::feed, "", {torch::arg("seq_id"), torch::arg("tokens")})
        // budget defaults to None: a decode budget that names none is unbounded.
        .def(
            "set_decode_budget",
            &pulsar::EngineBinding::set_decode_budget,
            "",
            {torch::arg("seq_id"), torch::arg("budget") = c10::IValue()}
        )
        .def("step", &pulsar::EngineBinding::step)
        .def("release", &pulsar::EngineBinding::release)
        .def("session_state", &pulsar::EngineBinding::session_state, "", {torch::arg("seq_id")})
        .def("demoted_tokens_count", &pulsar::EngineBinding::demoted_tokens_count)
        .def("active_tokens_count", &pulsar::EngineBinding::active_tokens_count)
        .def("stats", &pulsar::EngineBinding::stats, "", {torch::arg("seq_id")})
        .def("context_page_ids", &pulsar::EngineBinding::context_page_ids, "", {torch::arg("seq_id")})
        .def("context_pages", &pulsar::EngineBinding::context_pages, "", {torch::arg("seq_id")})
        .def("history_page_ids", &pulsar::EngineBinding::history_page_ids, "", {torch::arg("seq_id")})
        .def("history_pages", &pulsar::EngineBinding::history_pages, "", {torch::arg("seq_id")});
}
