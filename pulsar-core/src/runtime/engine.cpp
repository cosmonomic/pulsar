#include "pulsar/runtime/engine.hpp"

#include "pulsar/model/compiled_model.hpp"
#include "pulsar/runtime/kv/kv_memory.hpp"
#include "pulsar/runtime/kv/paged_bucket.hpp"
#include "pulsar/ops.hpp"  // write_kv_cuda, reposition_kv_cuda
#include "pulsar/profiling.hpp"
#include "pulsar/model/qwen2.hpp"
#include "pulsar/model/qwen3.hpp"

#include <ATen/Functions.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <ranges>
#include <set>
#include <vector>

namespace pulsar {

namespace {

// log of the keys each query row attended, the rerank's length reference. [T] fp32.
at::Tensor log_attended(const std::vector<int64_t>& attended_keys) {
    std::vector<float> out(attended_keys.size());
    for (size_t i = 0; i < attended_keys.size(); ++i) {
        out[i] = static_cast<float>(std::log(static_cast<double>(std::max<int64_t>(attended_keys[i], 1))));
    }
    return at::tensor(out, at::TensorOptions().dtype(at::kFloat));
}

// The model id is FAMILY (a prefix) plus an optional PRECISION suffix.
bool has_suffix(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}
bool has_prefix(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

std::unique_ptr<Model> build_model(const EngineConfig& c, int64_t num_pages) {
    const std::string& id = c.model;
    // PRECISION: a trailing -fp8/-int4 suffix is rejected; only bf16 is built.
    TORCH_CHECK(
        !has_suffix(id, "-fp8") && !has_suffix(id, "-int4"),
        "Engine: model '",
        id,
        "' requests an unsupported precision (only bf16 is built)"
    );
    if (has_prefix(id, "qwen3")) {
        return std::make_unique<Qwen3Model>(
            c.n_layers,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.hidden,
            c.intermediate,
            c.vocab,
            c.rope_theta,
            c.rms_eps,
            c.weights
        );
    }
    if (has_prefix(id, "qwen2")) {
        return std::make_unique<Qwen2Model>(
            c.n_layers,
            c.n_heads,
            c.n_kv_heads,
            c.head_dim,
            c.hidden,
            c.intermediate,
            c.vocab,
            c.rope_theta,
            c.rms_eps,
            c.weights
        );
    }
    if (id == "compiled") {
        return std::make_unique<CompiledModel>(c.decode_pt2, c.prefill_pt2, c.max_sessions, num_pages, c.page_size);
    }
    TORCH_CHECK(false, "Engine: unknown model '", id, "' (expected 'qwen3*', 'qwen2*', or 'compiled')");
}

RecallConfig build_recall_config(const EngineConfig& c, const std::vector<int64_t>& relevance_layers) {
    RecallConfig rc;
    // The score's denominator is the kernel's own LSE, so this must be the scale the
    // forward pass used.
    rc.scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
    rc.rope_theta = c.rope_theta;
    // The reranker's length gain must be the eviction density's, or the two sides of
    // the bar are in different units.
    rc.mass_reference_length = c.mass_reference_length;
    // A negative sampler seed means "random"; recall maps it to a fixed 0 so selection
    // stays reproducible.
    rc.seed = c.seed < 0 ? 0 : c.seed;
    rc.layers = relevance_layers;
    return rc;
}

}  // namespace

// The Scheduler holds a reference to the pool, so the pool must outlive it
// (header declaration order guarantees this).
Engine::Engine(EngineConfig config)
    : default_params{config.temperature, config.top_p, config.top_k, config.seed},
      eos_id(config.eos_id),
      device(at::Device(config.device)),
      active_buffer_size(config.active_buffer_size),
      max_chunk_size(config.max_chunk_size),
      max_sessions(config.max_sessions),
      page_size(config.page_size),
      active_max_size(std::max<int64_t>(0, config.active_buffer_size - config.max_chunk_size)),
      mass_reference_length(config.mass_reference_length),
      decode_cycle_interval(resolve_decode_cycle_interval(config.decode_cycle_interval, config.max_chunk_size)),
      default_session(config.session),
      recall_buffer_size(config.recall_buffer_size),
      ram_bucket_size(config.ram_bucket_size),
      recall_skip_fresh(config.recall_skip_fresh) {
    TORCH_CHECK(config.page_size > 0, "Engine: page_size must be > 0");
    TORCH_CHECK(
        config.active_buffer_size > 0 && config.active_buffer_size % config.page_size == 0,
        "Engine: active_buffer_size (",
        config.active_buffer_size,
        ") must be a positive multiple of page_size (",
        config.page_size,
        ")"
    );
    TORCH_CHECK(config.max_sessions > 0, "Engine: max_sessions must be > 0");
    // A parked session holds KV while not running, so pool capacity must cover running
    // PLUS parked. Catching this at startup beats running out of pages mid-run.
    TORCH_CHECK(
        config.max_running <= config.max_sessions,
        "Engine: max_running (",
        config.max_running,
        ") must not exceed max_sessions (",
        config.max_sessions,
        ")"
    );
    const int64_t num_pages = config.max_sessions * (config.active_buffer_size / config.page_size);
    // The compiled path bakes n_heads into the .pt2 and leaves it unset here, and never
    // evicts; falling back to n_kv_heads (group 1) still sizes a valid mass pool.
    const int64_t n_q_heads = config.n_heads > 0 ? config.n_heads : config.n_kv_heads;
    this->active = std::make_unique<ActiveBuffer>(
        config.n_layers,
        num_pages,
        config.page_size,
        config.n_kv_heads,
        n_q_heads,
        config.head_dim,
        config.dtype,
        config.device,
        config.rope_theta
    );
    std::vector<int64_t> relevance_layers = config.relevance_layers;
    if (relevance_layers.empty()) {
        for (int64_t l = 3 * config.n_layers / 4; l < config.n_layers; ++l) {
            relevance_layers.push_back(l);
        }
    }
    // With recall off nothing evicts, so no layer's mass is read and none is
    // accumulated; the recall config keeps the real list either way.
    this->active->set_relevance_layers(this->recall_buffer_size > 0 ? relevance_layers : std::vector<int64_t>{});
    this->active->set_mass_reference_length(config.mass_reference_length);
    this->model = build_model(config, num_pages);
    this->sched = std::make_unique<Scheduler>(*this->active, config.max_running, config.max_chunk_size);
    this->sampler = std::make_unique<StochasticSampler>();
    this->recall_cfg = build_recall_config(config, relevance_layers);
    this->configure_recall(config.spill_dir);
}

// Every token-valued knob is converted to the unit its site works in and both taus to a
// density bar, so a cycle reads numbers and never re-derives or re-parses one. A
// malformed value throws here, which is create().
SessionRuntime Engine::resolve_session(const SessionConfig& cfg) const {
    SessionRuntime rt;
    // No default: the protected window is the one knob whose right value follows from
    // the workload, so a session that recalls states it and a derived one would be a
    // guess nothing announces. Unread with recall off, where nothing evicts.
    TORCH_CHECK(
        this->recall_buffer_size <= 0 || cfg.n_working > 0,
        "session: n_working (",
        cfg.n_working,
        ") must be set explicitly in TOKENS when recall is on; it has no "
        "default"
    );
    rt.n_working = cfg.n_working;
    rt.query_window = cfg.recall_query_window;
    TORCH_CHECK(
        rt.query_window > 0,
        "session: recall_query_window (",
        rt.query_window,
        ") must be > 0; 1 reranks against the most recent token alone"
    );
    TORCH_CHECK(
        this->recall_buffer_size <= 0 || rt.query_window <= rt.n_working,
        "session: recall_query_window (",
        rt.query_window,
        ") must not exceed n_working (",
        rt.n_working,
        "), the window eviction protects"
    );
    rt.block_radius_pages = resolve_block_radius(cfg.block_radius, this->page_size);
    rt.position_layout = parse_position_layout(cfg.active_position_layout);
    rt.short_offset = cfg.short_offset;
    TORCH_CHECK(
        rt.short_offset >= 0,
        "session: short_offset (",
        rt.short_offset,
        ") must not be negative; 0 resolves to n_sink"
    );
    rt.segment_size = cfg.recall_segment_size > 0 ? cfg.recall_segment_size : this->page_size;
    TORCH_CHECK(
        rt.segment_size % this->page_size == 0,
        "session: recall_segment_size (",
        rt.segment_size,
        ") must be a multiple of page_size (",
        this->page_size,
        ")"
    );
    rt.prefilter_k = cfg.recall_prefilter_k;
    rt.bm25_query_tokens = this->recall_buffer_size > 0
        ? resolve_bm25_query_tokens(cfg.recall_bm25_query_tokens, rt.n_working)
        : 0;
    rt.recall_temperature = cfg.recall_temperature;
    // Same formula, same units, on each direction's own tau.
    rt.recall_bar = evict_bar_from_tau(cfg.recall_tau);
    rt.evict_bar = evict_bar_from_tau(cfg.evict_tau);
    TORCH_CHECK(
        this->recall_buffer_size <= 0 || this->active_max_size > 0,
        "session: active_buffer_size (",
        this->active_buffer_size,
        ") must exceed max_chunk_size (",
        this->max_chunk_size,
        ") when recall is on; the cycle has no room otherwise"
    );
    rt.attention_mass_decay = this->recall_buffer_size > 0
        ? resolve_mass_decay(cfg.attention_mass_decay, this->active_max_size)
        : 0.0;
    rt.recall_during_prefill = cfg.recall_during_prefill;
    return rt;
}

// recall_buffer_size <= 0 keeps the non-recall path (no demoted KV, no eviction); a
// configured buffer (this->memory != nullptr) is the runtime recall-on gate.
// Geometry derives from the active pool: n_kv_heads/head_dim from k_pool(0)'s shape,
// n_layers/page_size/dtype from the buffer itself.
void Engine::configure_recall(const std::string& spill_dir) {
    if (this->recall_buffer_size <= 0) {
        return;
    }
    at::Tensor kp = this->active->k_pool(0);
    int64_t n_kv_heads = kp.size(2);
    int64_t head_dim = kp.size(3);
    const int64_t ps = this->active->page_size();
    // The rerank scores the query's pre-RoPE Q, which only a native Qwen3 exposes.
    TORCH_CHECK(
        this->model->supports_capture(),
        "cascade recall requires a native model that exposes pre-RoPE Q "
        "(Qwen3); this model does not"
    );
    this->model->set_capture_layers(this->recall_cfg.layers);
    this->recall_buffer = std::make_unique<RecallBuffer>(
        static_cast<int64_t>(this->recall_cfg.layers.size()),
        n_kv_heads,
        head_dim,
        ps,
        kp.scalar_type(),
        this->recall_buffer_size / ps,
        this->max_sessions,
        kp.device()
    );
    // Only S2 can absorb S1's spill, so a budget with nowhere to drain to would fill its
    // reserve and die mid-run.
    TORCH_CHECK(
        this->ram_bucket_size <= 0 || !spill_dir.empty(),
        "Engine: ram_bucket_size (",
        this->ram_bucket_size,
        ") needs a spill_dir to drain into; set spill_dir or leave "
        "ram_bucket_size at 0 for an unbudgeted host bucket"
    );
    this->s1 = std::make_unique<PagedBucket>(
        this->active->n_layers(),
        n_kv_heads,
        head_dim,
        ps,
        kp.scalar_type(),
        at::kCPU,
        this->ram_bucket_size > 0 ? this->ram_bucket_size / ps : 0,
        this->active_buffer_size / ps
    );
    // S2: overflow behind S1, disabled (no files) when spill_dir is empty.
    this->s2 = std::make_unique<FileBucket>(
        this->active->n_layers(),
        n_kv_heads,
        head_dim,
        ps,
        kp.scalar_type(),
        spill_dir
    );
    this->memory = std::make_unique<KVMemory>(
        *this->active,
        *this->recall_buffer,
        *this->s1,
        *this->s2,
        this->recall_cfg,
        this->recall_skip_fresh
    );
}

int64_t Engine::create_session(c10::List<int64_t> system_tokens, const SessionConfig& overrides) {
    TORCH_CHECK(system_tokens.size() > 0, "create_session: system prompt must be non-empty");
    // Monotonic and never reused, so a session this engine has ever opened can never
    // collide with one still live.
    const int64_t seq_id = this->next_seq_id++;
    // Resolve BEFORE the prefill is queued: the cycle reads the resolved knobs off the
    // sequence, and a malformed one must fail with nothing enqueued.
    this->open_session(seq_id, std::move(system_tokens), this->resolve_session(overrides));
    return seq_id;
}

// Fresh session: this feed is the system prompt, which becomes the sink.
void Engine::open_session(int64_t seq_id, c10::List<int64_t> tokens, const SessionRuntime& cfg) {
    // A session may grow to a whole window and the pool holds exactly max_sessions of
    // them, so admitting more would leave a sequence waiting for capacity nothing will
    // free. One entry here is one live session whatever state it is in: a continued
    // session is queued AND holding KV, and counts once.
    const int64_t holding = std::ssize(this->seqs);
    TORCH_CHECK(
        holding < this->max_sessions,
        "session: seq ",
        seq_id,
        " would be session ",
        holding + 1,
        " of max_sessions ",
        this->max_sessions,
        "; release a session first"
    );
    const int64_t n = static_cast<int64_t>(tokens.size());
    SeqState st;
    st.cfg = cfg;
    st.pending.reserve(n);
    for (size_t i = 0; i < tokens.size(); ++i) {
        st.pending.push_back(tokens.get(i));
    }
    st.pending_base = 0;  // prompt sits at addresses [0, n)
    st.stream_len = n;
    st.prompt_len = n;  // generation output begins after the prompt
    st.n_sink = n;  // the system prompt is the protected sink
    // The 0 sentinel resolves to n_sink, this feed's length.
    TORCH_CHECK(
        st.cfg.short_offset == 0 || st.cfg.short_offset >= n,
        "session: short_offset (",
        st.cfg.short_offset,
        ") is inside seq ",
        seq_id,
        "'s protected sink of ",
        n,
        " tokens; it must be at or after the sink"
    );
    st.short_offset = st.cfg.short_offset > 0 ? st.cfg.short_offset : n;
    st.params = this->default_params;
    st.gen = make_generator(this->device, st.params.seed);
    st.decode_since_cycle = 0;
    st.stats = {};
    this->seqs.emplace(seq_id, std::move(st));
    if (this->memory) {
        this->s1->begin(seq_id);
        this->s2->begin(seq_id);  // no-op when the disk bucket is disabled
    }
    // Not scheduled here: the system prompt alone is a complete, generatable prefill,
    // and a caller driving step() continuously (not just when it has fed a real turn)
    // would otherwise generate off the sink by itself. feed()'s first call schedules it.
}

void Engine::feed(int64_t seq_id, c10::List<int64_t> tokens) {
    auto it = this->seqs.find(seq_id);
    if (it == this->seqs.end()) {
        if (tokens.size() == 0) {
            return;
        }
        // A session opened by feed() alone names no overrides, so it takes the engine's
        // defaults. This feed IS the first real turn, so it is scheduled immediately
        // (contrast open_session(), called from create_session(), which is not).
        this->open_session(seq_id, std::move(tokens), this->resolve_session(this->default_session));
        SeqState& fresh = this->seqs.at(seq_id);
        this->sched->add_request(seq_id, fresh.stream_len);
        return;
    }
    SeqState& st = it->second;
    if (!this->active->has_seq(seq_id)) {
        // Opened but not yet prefilled: extend the initial prefill (the following
        // content turn after create). n_sink stays the system-prompt length. Not yet
        // waiting means this is the session's first feed since create_session() (which
        // does not schedule it); otherwise a still-pending prior feed already did.
        TORCH_CHECK(!this->sched->is_running(seq_id), "feed: seq ", seq_id, " has started prefill; cannot extend");
        const int64_t added = static_cast<int64_t>(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            st.pending.push_back(tokens.get(i));
        }
        st.stream_len += added;
        st.prompt_len += added;  // still the first turn's prompt
        if (this->sched->is_waiting(seq_id)) {
            this->sched->grow_request(seq_id, st.stream_len);
        } else {
            this->sched->add_request(seq_id, st.stream_len);
        }
        return;
    }
    // A prefill that has started but not landed still holds its remaining ids in the
    // pending buffer. Continuing over it would park the sequence (discarding the
    // scheduler's prefill cursor), leave the addresses below the tail never prefilled,
    // and lead the pending buffer with last_token == -1, which reaches the embedding as
    // an out-of-range index.
    TORCH_CHECK(
        st.pending.empty(),
        "feed: seq ",
        seq_id,
        " is mid-prefill (",
        st.pending.size(),
        " tokens pending); step() it to completion before feeding again"
    );
    // Continuation over active KV: a still-running sequence is parked here, which
    // keeps its KV active. Append the tokens and re-admit; resume re-prefills
    // addresses [pending_base, stream_len) over the active KV, at the pool tail. The
    // 1-token lag from the prior turn (last_token, its KV never written) leads the
    // pending buffer at address old_stream - 1; the appended turn follows. The
    // re-prefill sources every id from the pending buffer (pending_base is the
    // re-prefill start).
    if (this->sched->is_running(seq_id)) {
        this->sched->park(seq_id);
    }
    const int64_t old_stream = st.stream_len;
    st.pending.clear();
    st.pending.push_back(st.last_token);  // lag token at address old_stream - 1
    for (size_t i = 0; i < tokens.size(); ++i) {
        st.pending.push_back(tokens.get(i));
    }
    st.pending_base = old_stream - 1;
    st.stream_len = old_stream + static_cast<int64_t>(tokens.size());
    st.prompt_len = st.stream_len;  // this turn's prompt ends here; the first
    // token emits only once its tail is prefilled
    st.decode_since_cycle = 0;
    st.stats = {};
    this->q_rings.erase(seq_id);  // last turn's decode Q is stale
    this->sched->resume(seq_id, st.pending_base, st.stream_len);
}

void Engine::set_decode_budget(int64_t seq_id, std::optional<int64_t> budget) {
    TORCH_CHECK(this->seqs.count(seq_id), "set_decode_budget: unknown seq ", seq_id);
    this->seqs.at(seq_id).budget = budget.value_or(-1);
}

int64_t Engine::active_sequence_count() const {
    return this->sched->num_running() + this->sched->num_waiting();
}

Engine::SessionState Engine::session_state(int64_t seq_id) const {
    if (!this->seqs.count(seq_id)) {
        return SessionState::unknown;
    }
    if (this->sched->is_running(seq_id) || this->sched->is_waiting(seq_id)) {
        return SessionState::active;
    }
    return SessionState::parked;
}

void Engine::release(int64_t seq_id) {
    // Drops it from waiting/running and frees its pages; a no-op once parked, whose
    // pages the explicit free below reclaims.
    this->sched->cancel(seq_id);
    if (this->active->has_seq(seq_id)) {
        this->active->free(seq_id);
    }
    if (this->memory) {
        this->s1->end(seq_id);
        this->s2->end(seq_id);
        this->memory->end_seq(seq_id);
    }
    this->q_rings.erase(seq_id);
    this->seqs.erase(seq_id);
}

int64_t Engine::demoted_tokens_count(int64_t seq_id) const {
    return this->memory ? this->memory->demoted_tokens(seq_id) : 0;
}

int64_t Engine::active_tokens_count(int64_t seq_id) const {
    return this->active->has_seq(seq_id) ? this->active->active_len(seq_id) : 0;
}

Stats Engine::stats(int64_t seq_id) const {
    auto it = this->seqs.find(seq_id);
    TORCH_CHECK(it != this->seqs.end(), "stats: unknown seq ", seq_id);
    return it->second.stats;
}

// Reconstruct token ids for addresses [start, end) without a stored stream: the
// pending buffer serves this turn's un-prefilled tail, the KV memory's segment
// documents serve prefilled/generated addresses below it.
std::vector<int64_t> Engine::token_ids_at(int64_t seq_id, int64_t start, int64_t end) {
    SeqState& st = this->seqs.at(seq_id);
    std::vector<int64_t> out;
    if (end <= start) {
        return out;
    }
    out.reserve(end - start);
    const int64_t pend_lo = st.pending_base;
    const int64_t pend_hi = st.pending_base + static_cast<int64_t>(st.pending.size());
    int64_t a = start;
    // Cached prefix: addresses below the pending buffer come from the cache.
    const int64_t cache_hi = st.pending.empty() ? end : std::min(end, pend_lo);
    if (cache_hi > a) {
        TORCH_CHECK(this->memory, "token_ids_at: cached addresses require recall");
        std::vector<int64_t> cached = this->memory->token_ids_in(seq_id, a, cache_hi);
        out.insert(out.end(), cached.begin(), cached.end());
        a = cache_hi;
    }
    for (; a < end; ++a) {
        TORCH_CHECK(a >= pend_lo && a < pend_hi, "token_ids_at: address ", a, " not in cache or pending");
        out.push_back(st.pending[a - pend_lo]);
    }
    return out;
}

// context = the active pages (the attention window the model sees) in address order.
// Works with recall on or off. *_page_ids returns the page indices; *_pages returns
// each page's token ids.
std::vector<int64_t> Engine::context_page_ids(int64_t seq_id) const {
    std::vector<int64_t> out;
    ActiveView v0{this->active.get(), seq_id};
    for (auto c = v0.lower_bound(0), e = v0.end(); !(c == e); ++c) {
        out.push_back((*c).first);
    }
    return out;
}

std::vector<std::vector<int64_t>> Engine::context_pages(int64_t seq_id) const {
    std::vector<std::vector<int64_t>> out;
    ActiveView v0{this->active.get(), seq_id};
    for (auto c = v0.lower_bound(0), e = v0.end(); !(c == e); ++c) {
        out.push_back((*c).second.token_ids());
    }
    return out;
}

// history = the whole conversation merged across the active buffer and the bucket
// chain, in address order; requires recall (the KV memory).
std::vector<int64_t> Engine::history_page_ids(int64_t seq_id) const {
    TORCH_CHECK(this->memory, "history_page_ids: requires recall (the KV memory)");
    std::vector<int64_t> out;
    PageView h = this->memory->history(seq_id);
    for (int64_t k : h | std::views::keys) {
        out.push_back(k);
    }
    return out;
}

std::vector<std::vector<int64_t>> Engine::history_pages(int64_t seq_id) const {
    TORCH_CHECK(this->memory, "history_pages: requires recall (the KV memory)");
    std::vector<std::vector<int64_t>> out;
    PageView h = this->memory->history(seq_id);
    for (const PageRef& p : h | std::views::values) {
        out.push_back(p.token_ids());
    }
    return out;
}

std::optional<GroupBatch> Engine::build_prefill(const BatchDescriptor& desc, AcceptPlan& accept) {
    const c10::List<int64_t>& pre_ids = desc.prefill_seq_ids;
    if (pre_ids.empty()) {
        return std::nullopt;
    }
    at::Tensor cu = desc.prefill_cu_seqlens_q.to(at::kCPU, at::kLong).contiguous();
    at::Tensor sk = desc.prefill_seqlens_k.to(at::kCPU, at::kLong).contiguous();
    at::Tensor ps = desc.prefill_prompt_starts.to(at::kCPU, at::kLong).contiguous();
    auto cu_a = cu.accessor<int64_t, 1>();
    auto sk_a = sk.accessor<int64_t, 1>();
    auto ps_a = ps.accessor<int64_t, 1>();
    std::vector<int64_t> input_ids, pos;
    // One EMA rate per sequence, in the group's row order: the kernels index it by the
    // sequence a query token belongs to.
    std::vector<float> mass_decay;
    for (size_t si = 0; si < pre_ids.size(); ++si) {
        int64_t sid = pre_ids.get(si);
        SeqState& st = this->seqs.at(sid);
        mass_decay.push_back(static_cast<float>(st.cfg.attention_mass_decay));
        const int64_t chunk = cu_a[si + 1] - cu_a[si];
        const int64_t end_pos = sk_a[si];  // true context tail after chunk
        const int64_t pos_start = end_pos - chunk;  // RoPE pos start (may exceed prompt index)
        const int64_t prompt_start = ps_a[si];  // conversation-address start
        // Token identity is read by conversation ADDRESS (== addr); the RoPE pos
        // is the pool tail. Source the ids from the pending buffer / the cache.
        std::vector<int64_t> chunk_ids = this->token_ids_at(sid, prompt_start, prompt_start + chunk);
        std::vector<int64_t> addr, addr_tok;
        addr.reserve(chunk);
        addr_tok.reserve(chunk);
        for (int64_t t = 0; t < chunk; ++t) {
            input_ids.push_back(chunk_ids[t]);
            // RoPE takes the buffer's layout position; pos_start + t is the SLOT index,
            // which set_identity and seqlens_k below still want. The two coincide only
            // when the buffer ropes contiguously.
            pos.push_back(this->active->rope_pos_at(sid, pos_start + t));
            addr.push_back(prompt_start + t);
            addr_tok.push_back(chunk_ids[t]);
        }
        // The active buffer owns page-index/token-id identity: record the chunk at
        // its pool tail. The segment documents record the same tokens by address as
        // they are born, independent of where their KV ends up living.
        this->active->set_identity(sid, pos_start, addr, addr_tok);
        if (this->memory) {
            this->memory->note_tokens(sid, addr, addr_tok, st.cfg.segment_size);
        }
        // Once this turn's prefill has fully landed, drop the pending buffer (its
        // ids now live in the cache).
        if (prompt_start + chunk >= st.stream_len) {
            st.pending.clear();
            st.pending_base = 0;
        }
        // Emit only when the PROMPT is fully prefilled (prompt progress, not the
        // tail: recall recompaction moves the pool tail off the prompt index).
        if (prompt_start + chunk >= st.prompt_len) {
            accept.emplace_back(sid, cu_a[si + 1] - 1);
        }
    }
    GroupBatch g;
    {
        at::Tensor t = at::empty({static_cast<int64_t>(input_ids.size())}, at::TensorOptions().dtype(at::kLong));
        if (!input_ids.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), input_ids.data(), input_ids.size() * sizeof(int64_t));
        }
        g.tokens = t.to(this->device);
    }
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pos.size())}, at::TensorOptions().dtype(at::kLong));
        if (!pos.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), pos.data(), pos.size() * sizeof(int64_t));
        }
        g.pos = t.to(this->device);
    }
    g.slot_mapping = desc.prefill_slot_mapping;
    g.page_tables = desc.prefill_page_tables;
    g.cu_seqlens_q = desc.prefill_cu_seqlens_q;
    g.seqlens_k = desc.prefill_seqlens_k;
    g.attention_mass_decay = at::tensor(mass_decay, at::TensorOptions().dtype(at::kFloat)).to(this->device);
    g.mass_length_gain = this->mass_length_gain();
    return g;
}

std::optional<GroupBatch> Engine::build_decode(const BatchDescriptor& desc, AcceptPlan& accept) {
    const c10::List<int64_t>& dec_ids = desc.decode_seq_ids;
    if (dec_ids.empty()) {
        return std::nullopt;
    }
    at::Tensor cl = desc.decode_context_lens.to(at::kCPU, at::kLong).contiguous();
    auto cl_a = cl.accessor<int64_t, 1>();
    std::vector<int64_t> input_ids, pos;
    // One EMA rate per sequence, in the group's row order (one query token each).
    std::vector<float> mass_decay;
    for (size_t si = 0; si < dec_ids.size(); ++si) {
        int64_t sid = dec_ids.get(si);
        const SeqState& st = this->seqs.at(sid);
        mass_decay.push_back(static_cast<float>(st.cfg.attention_mass_decay));
        input_ids.push_back(st.last_token);
        // Slot index cl_a[si] - 1 for identity below, layout position for RoPE.
        pos.push_back(this->active->rope_pos_at(sid, cl_a[si] - 1));
        // The token written into the pool this step is the previously sampled
        // last_token; its address is the last conversation position (stream_len -
        // 1). It lands at the pool tail, pool pos cl_a[si] - 1.
        const int64_t addr = st.stream_len - 1;
        this->active->set_identity(sid, cl_a[si] - 1, {addr}, {st.last_token});
        if (this->memory) {
            this->memory->note_tokens(sid, {addr}, {st.last_token}, st.cfg.segment_size);
        }
        accept.emplace_back(sid, static_cast<int64_t>(si));  // every decode emits
    }
    GroupBatch g;
    {
        at::Tensor t = at::empty({static_cast<int64_t>(input_ids.size())}, at::TensorOptions().dtype(at::kLong));
        if (!input_ids.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), input_ids.data(), input_ids.size() * sizeof(int64_t));
        }
        g.tokens = t.to(this->device);
    }
    {
        at::Tensor t = at::empty({static_cast<int64_t>(pos.size())}, at::TensorOptions().dtype(at::kLong));
        if (!pos.empty()) {
            std::memcpy(t.data_ptr<int64_t>(), pos.data(), pos.size() * sizeof(int64_t));
        }
        g.pos = t.to(this->device);
    }
    g.slot_mapping = desc.decode_slot_mapping;
    g.page_tables = desc.decode_page_tables;
    g.context_lens = desc.decode_context_lens;
    g.attention_mass_decay = at::tensor(mass_decay, at::TensorOptions().dtype(at::kFloat)).to(this->device);
    g.mass_length_gain = this->mass_length_gain();
    return g;
}

void Engine::accept_group(
    const at::Tensor& logits,
    const AcceptPlan& accept,
    std::vector<int64_t>& ids,
    std::vector<int64_t>& tokens,
    std::vector<bool>& finished
) {
    for (const auto& [sid, row] : accept) {
        SeqState& st = this->seqs.at(sid);
        int64_t token = this->sampler->sample(logits.select(0, row), st.params, st.gen);
        // The sampled token joins the conversation (a new address) and this turn's
        // generated output; its KV is written next step (decode input).
        st.last_token = token;
        st.stream_len += 1;
        if (st.budget > 0) {
            st.budget -= 1;
        }
        ids.push_back(sid);
        tokens.push_back(token);
        finished.push_back(this->finished(st, token));
    }
}

bool Engine::finished(const SeqState& st, int64_t token) const {
    return (this->eos_id >= 0 && token == this->eos_id) || st.budget == 0;
}

std::vector<TokenEvent> Engine::step_once() {
    PULSAR_PROF_SCOPE(STEP);
    BatchDescriptor desc = this->sched->step();
    AcceptPlan pre_accept, dec_accept;
    std::optional<GroupBatch> prefill = this->build_prefill(desc, pre_accept);
    std::optional<GroupBatch> decode = this->build_decode(desc, dec_accept);
    // Attention-mass EMA upkeep, BEFORE the attention kernels add this step's received
    // mass: m_new = (1 - decay)^n_tokens * m_old + received. A decode forwards one token,
    // a prefill its chunk. Advancing forwarded_tokens by the same count keeps the spill's
    // lazy decay on the same per-token clock.
    for (size_t i = 0; i < desc.decode_seq_ids.size(); ++i) {
        const int64_t sid = desc.decode_seq_ids.get(i);
        SeqState& st = this->seqs.at(sid);
        this->active->upkeep(sid, 1, st.cfg.attention_mass_decay);
        st.forwarded_tokens += 1;
    }
    if (!desc.prefill_seq_ids.empty()) {
        at::Tensor cu = desc.prefill_cu_seqlens_q.to(at::kCPU, at::kInt).contiguous();
        auto qa = cu.accessor<int32_t, 1>();
        for (size_t i = 0; i < desc.prefill_seq_ids.size(); ++i) {
            const int64_t sid = desc.prefill_seq_ids.get(i);
            const int64_t chunk = static_cast<int64_t>(qa[i + 1] - qa[i]);
            SeqState& st = this->seqs.at(sid);
            this->active->upkeep(sid, chunk, st.cfg.attention_mass_decay);
            st.forwarded_tokens += chunk;
        }
    }
    StepLogits logits = this->model->forward(StepBatch{prefill, decode}, *this->active);
    // Append this step's decode-group Q to each decode seq's ring (the rerank's
    // query window); no-op with recall off and on a prefill-only step.
    this->update_q_ring(desc);
    std::vector<int64_t> ids, tokens;
    std::vector<bool> fin;
    if (prefill.has_value()) {
        this->accept_group(logits.prefill.value(), pre_accept, ids, tokens, fin);
    }
    if (decode.has_value()) {
        this->accept_group(logits.decode.value(), dec_accept, ids, tokens, fin);
    }
    // Every sequence that finished this step is PARKED (KV kept active), never
    // freed. Still-generating emissions go through update() (finished=false);
    // finishers are parked afterward.
    c10::List<int64_t> upd_ids, upd_tokens;
    c10::List<bool> upd_finished;
    std::vector<int64_t> park_ids;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (fin[i]) {
            park_ids.push_back(ids[i]);
            continue;
        }
        upd_ids.push_back(ids[i]);
        upd_tokens.push_back(tokens[i]);
        upd_finished.push_back(fin[i]);
    }
    this->sched->update(upd_ids, upd_tokens, upd_finished);
    for (int64_t sid : park_ids) {
        this->sched->park(sid);
    }
    if (this->memory) {
        this->recall_cycle(desc);
    }
    std::vector<TokenEvent> events;
    events.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        events.push_back(TokenEvent{ids[i], tokens[i], fin[i]});
    }
    return events;
}

std::vector<TokenEvent> Engine::step(int64_t n) {
    TORCH_CHECK(n >= 1, "step: n must be >= 1");
    std::vector<TokenEvent> events;
    while (std::ssize(events) < n && this->active_sequence_count() > 0) {
        std::vector<TokenEvent> emitted = this->step_once();
        events.insert(events.end(), emitted.begin(), emitted.end());
        // feed() caps live sessions at max_sessions and the pool is exactly that many
        // windows, so a waiting sequence is always admittable. Were it not, nothing
        // would run and nothing would free capacity, and this would spin forever.
        TORCH_CHECK(
            !emitted.empty() || this->sched->num_running() > 0,
            "step: no sequence is running and the pool cannot admit the ",
            this->sched->num_waiting(),
            " waiting"
        );
    }
    return events;
}

void Engine::evict_demote(int64_t seq_id, bool decode) {
    const int64_t n = this->active->active_len(seq_id);
    SeqState& st = this->seqs.at(seq_id);
    const int64_t n_sink = st.n_sink;
    TORCH_CHECK(
        n_sink + st.cfg.n_working < this->active_max_size,
        "seq ",
        seq_id,
        ": protected sink (",
        n_sink,
        ") plus working window (",
        st.cfg.n_working,
        ") leaves no evictable band under active_max_size ",
        this->active_max_size
    );
    if (n <= n_sink + st.cfg.n_working) {
        return;  // no evictable band yet
    }
    // Demote every evictable page whose NEIGHBOURHOOD bias-corrected density is below
    // the keep-bar, with active_max_size as a backstop.
    std::vector<float> cand_density;
    EvictAttribution attribution;
    DemotedKV demoted = PULSAR_PROF_EXPR(
        EVICT,
        this->active->evict(
            seq_id,
            n_sink,
            st.cfg.n_working,
            st.cfg.position_layout,
            st.short_offset,
            st.cfg.evict_bar,
            this->active_max_size,
            st.cfg.block_radius_pages,
            st.stream_len,
            st.cfg.attention_mass_decay,
            &cand_density,
            &st.stats.density_by_position,
            &attribution
        )
    );
    // One mass entry per evicted page.
    st.stats.evicted_pages += demoted.page_mass().numel();
    for (float d : cand_density) {
        ++st.stats.density_hist[density_bin(d)];
    }
    Stats& stats = st.stats;
    stats.bar_evicted_pages += attribution.bar_pages;
    stats.backstop_evicted_pages += attribution.backstop_pages;
    stats.raw_mass_victim_pages += attribution.raw_mass_victims;
    stats.raw_mass_agree_pages += attribution.raw_mass_agree;
    for (int64_t age : attribution.candidate_age) {
        ++stats.candidate_age_hist[age_bin(static_cast<double>(age))];
    }
    for (int64_t age : attribution.victim_age) {
        ++stats.evicted_age_hist[age_bin(static_cast<double>(age))];
    }
    if (decode) {
        stats.bar_evicted_pages_decode += attribution.bar_pages;
        stats.backstop_evicted_pages_decode += attribution.backstop_pages;
        stats.raw_mass_victim_pages_decode += attribution.raw_mass_victims;
        stats.raw_mass_agree_pages_decode += attribution.raw_mass_agree;
        for (int64_t age : attribution.candidate_age) {
            ++stats.candidate_age_hist_decode[age_bin(static_cast<double>(age))];
        }
        for (int64_t age : attribution.victim_age) {
            ++stats.evicted_age_hist_decode[age_bin(static_cast<double>(age))];
        }
    }
    // The victims move STRAIGHT into s1 (a D2H): reserve the descriptors, then fill.
    PULSAR_PROF_EXPR(ONEVICT, this->memory->accept_evicted(seq_id, demoted, st.forwarded_tokens));
}

// One writer window with no live reader: after promote, return the slots both
// buckets marked dead this cycle and spill S1's overflow into S2.
void Engine::end_cycle(int64_t seq_id) {
    const SeqState& st = this->seqs.at(seq_id);
    this->memory->end_cycle(seq_id, this->ram_bucket_size, st.forwarded_tokens, st.cfg.attention_mass_decay);
}

std::vector<at::Tensor> Engine::gather_query_q(int64_t row_lo, int64_t row_hi) const {
    // The rerank scores on the GPU, so the query Q stays on device.
    std::vector<at::Tensor> out;
    out.reserve(this->recall_cfg.layers.size());
    for (int64_t layer : this->recall_cfg.layers) {
        at::Tensor q = this->model->captured_q(/*is_prefill=*/true, layer);
        TORCH_CHECK(q.defined(), "gather_query_q: layer ", layer, " not captured");
        out.push_back(q.slice(0, row_lo, row_hi).contiguous());
    }
    return out;
}

std::vector<at::Tensor> Engine::gather_query_lse(int64_t row_lo, int64_t row_hi) const {
    std::vector<at::Tensor> out;
    out.reserve(this->recall_cfg.layers.size());
    for (int64_t layer : this->recall_cfg.layers) {
        at::Tensor lse = this->model->captured_lse(/*is_prefill=*/true, layer);
        TORCH_CHECK(lse.defined(), "gather_query_lse: layer ", layer, " not captured");
        out.push_back(lse.slice(0, row_lo, row_hi).contiguous());
    }
    return out;
}

// Append this step's decode-group Q to each decode seq's per-layer ring. Each
// decode seq forwards exactly one token, at decode-group row si (build_decode's
// order == desc.decode_seq_ids), so captured_q(false, l)[si] is that seq's
// just-decoded token Q. Rings are lazily allocated on first write from the
// captured shape, sized to that sequence's own query window, on the model
// device/dtype.
void Engine::update_q_ring(const BatchDescriptor& desc) {
    if (!this->memory) {
        return;  // recall off: no capture, no ring
    }
    const c10::List<int64_t>& dec_ids = desc.decode_seq_ids;
    if (dec_ids.empty()) {
        return;
    }
    const std::vector<int64_t>& layers = this->recall_cfg.layers;
    const int64_t nl = static_cast<int64_t>(layers.size());
    // Captured Q per layer for the decode group, [ntok_decode, n_heads, head_dim].
    std::vector<at::Tensor> qcap(nl);
    for (int64_t li = 0; li < nl; ++li) {
        qcap[li] = this->model->captured_q(false, layers[li]);
        TORCH_CHECK(qcap[li].defined(), "update_q_ring: layer ", layers[li], " not captured on the decode group");
    }
    for (size_t si = 0; si < dec_ids.size(); ++si) {
        const int64_t sid = dec_ids.get(si);
        const int64_t window = this->seqs.at(sid).cfg.query_window;
        QRing& ring = this->q_rings[sid];
        if (ring.buf.empty()) {  // lazy allocation from the captured shape
            ring.buf.resize(nl);
            ring.lse.resize(nl);
            ring.ctx.assign(window, 0);
            for (int64_t li = 0; li < nl; ++li) {
                const at::Tensor& q = qcap[li];
                ring.buf[li] = at::empty({window, q.size(1), q.size(2)}, q.options());
                ring.lse[li] = at::empty({window, q.size(1)}, q.options().dtype(at::kFloat));
            }
        }
        const int64_t slot = ring.cursor;
        for (int64_t li = 0; li < nl; ++li) {
            ring.buf[li].select(0, slot).copy_(qcap[li].select(0, static_cast<int64_t>(si)));
            at::Tensor lcap = this->model->captured_lse(false, layers[li]);
            TORCH_CHECK(
                lcap.defined(),
                "update_q_ring: layer ",
                layers[li],
                " has no captured lse on the decode group"
            );
            ring.lse[li].select(0, slot).copy_(lcap.select(0, static_cast<int64_t>(si)));
        }
        // The token attended the whole active context as it stood this step.
        ring.ctx[slot] = this->active->active_len(sid);
        ring.cursor = (slot + 1) % window;
        if (ring.count < window) {
            ++ring.count;
        }
    }
}

std::vector<int64_t> Engine::QRing::chronological(int64_t window) const {
    const int64_t n = std::min(this->count, window);
    const int64_t start = ((this->cursor - n) % window + window) % window;
    std::vector<int64_t> slots(n);
    for (int64_t i = 0; i < n; ++i) {
        slots[i] = (start + i) % window;
    }
    return slots;
}

Engine::RecallQuery Engine::read_q_ring(int64_t seq_id) const {
    auto it = this->q_rings.find(seq_id);
    if (it == this->q_rings.end()) {
        return {};
    }
    const QRing& ring = it->second;
    if (ring.count == 0 || ring.buf.empty()) {
        return {};
    }
    const std::vector<int64_t> slots = ring.chronological(this->seqs.at(seq_id).cfg.query_window);
    at::Tensor order = at::tensor(slots, at::TensorOptions().dtype(at::kLong)).to(this->device);
    RecallQuery query;
    query.vecs.reserve(ring.buf.size());
    query.lse.reserve(ring.lse.size());
    for (const at::Tensor& q : ring.buf) {
        query.vecs.push_back(q.index_select(0, order).contiguous());
    }
    for (const at::Tensor& l : ring.lse) {
        query.lse.push_back(l.index_select(0, order).contiguous());
    }
    std::vector<int64_t> attended;
    attended.reserve(slots.size());
    for (int64_t slot : slots) {
        attended.push_back(ring.ctx[slot]);
    }
    query.log_attended_len = log_attended(attended);
    return query;
}

at::Tensor Engine::query_row_positions(int64_t seq_id, const RecallQuery& query) const {
    const int64_t rows = query.vecs.empty() ? 0 : query.vecs[0].size(0);
    if (rows == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kLong));
    }
    const int64_t active_len = this->active->active_len(seq_id);
    TORCH_CHECK(
        rows <= active_len,
        "query_row_positions: seq ",
        seq_id,
        " has ",
        active_len,
        " active tokens, fewer than the ",
        rows,
        " query rows"
    );
    std::vector<int64_t> positions(rows);
    for (int64_t i = 0; i < rows; ++i) {
        positions[i] = this->active->rope_pos_at(seq_id, active_len - rows + i);
    }
    return at::tensor(positions, at::TensorOptions().dtype(at::kLong));
}

int64_t Engine::candidate_scoring_position(int64_t seq_id) const {
    const SeqState& st = this->seqs.at(seq_id);
    if (st.cfg.position_layout == PositionLayout::compacted) {
        return st.short_offset;
    }
    return ActiveBuffer::working_start(this->active->active_len(seq_id), st.n_sink, st.cfg.n_working) - 1;
}

void Engine::recall_recompact(int64_t seq_id, const RecallQuery& query, bool decode) {
    // Promote at most what active_max_size leaves free. Selection stops itself on that
    // capacity.
    const SessionRuntime& cfg = this->seqs.at(seq_id).cfg;
    const int64_t short_offset = this->seqs.at(seq_id).short_offset;
    const int64_t capacity_tokens = this->active_max_size - this->active->active_len(seq_id);
    if (capacity_tokens <= 0) {
        return;
    }
    at::Tensor q = at::tensor(query.token_ids, at::TensorOptions().dtype(at::kLong));
    std::vector<double> raw_scores;
    RecallCounts counts;
    const RecallParams params{
        cfg.recall_bar,
        cfg.recall_temperature,
        cfg.prefilter_k,
        short_offset,
        this->candidate_scoring_position(seq_id),
        cfg.segment_size,
        cfg.block_radius_pages
    };
    std::vector<Payload> payloads = PULSAR_PROF_EXPR(
        MLC_RECALL,
        this->memory->recall(
            seq_id,
            q,
            capacity_tokens,
            query.vecs,
            query.lse,
            query.log_attended_len,
            this->query_row_positions(seq_id, query),
            params,
            &raw_scores,
            &counts
        )
    );
    // Bin the scores as scored, before the empty-payload early-out, so the histogram
    // reflects the whole scored distribution however selection went. raw_scores are
    // densities, already multiples of uniform; recall_hist squashes each to r / (1 + r)
    // so the bins keep parity with tau at 0.5.
    Stats& stats = this->seqs.at(seq_id).stats;
    for (double r : raw_scores) {
        // -inf marks a page that was never scored; NaN is not a score. An overflowed
        // score is a real one, and squashes to the top bin.
        if (std::isnan(r) || r == -std::numeric_limits<double>::infinity()) {
            continue;
        }
        const double s = std::isinf(r) ? 1.0 : r / (1.0 + r);
        ++stats.recall_hist[recall_bin(s)];
        if (decode) {
            ++stats.recall_hist_decode[recall_bin(s)];
        }
    }
    stats.staged_hits += counts.staged_hits;
    stats.staged_misses += counts.staged_misses;
    stats.promoted_s1 += counts.promoted_s1;
    stats.promoted_s2 += counts.promoted_s2;
    if (payloads.empty()) {
        return;  // nothing recalled; survivors already compacted
    }
    // Count PAGES actually promoted this cycle (one Payload per page), not candidates.
    stats.recalled_pages += static_cast<int64_t>(payloads.size());

    const int64_t n_layers = this->active->n_layers();
    // Survivors: the post-evict_demote active set at logical [0, K), addr ascending.
    // Read their identity (page_id/tok) in page-position order: page pp holds addresses
    // [page_id*ps, ...), so surv_addr is ascending and == active_len.
    const int64_t K = this->active->active_len(seq_id);
    const int64_t ps = this->active->page_size();
    std::vector<int64_t> surv_addr, surv_tok;
    surv_addr.reserve(K);
    surv_tok.reserve(K);
    const int64_t np = this->active->num_active_pages(seq_id);
    for (int64_t pp = 0; pp < np; ++pp) {
        const int64_t base = this->active->page_id(seq_id, pp) * ps;
        const std::vector<int64_t>& ids = this->active->page_token_ids(seq_id, pp);
        for (int64_t o = 0; o < static_cast<int64_t>(ids.size()); ++o) {
            surv_addr.push_back(base + o);
            surv_tok.push_back(ids[o]);
        }
    }
    TORCH_CHECK(
        static_cast<int64_t>(surv_addr.size()) == K,
        "recall_recompact: survivor count ",
        surv_addr.size(),
        " != active_len ",
        K
    );

    // Flatten the recalled payloads' identity into one recalled set (row r spans
    // payloads in order). A page's addresses are contiguous from its base.
    std::vector<at::Tensor> tok_parts;
    std::vector<int64_t> rec_addr;
    for (const Payload& pl : payloads) {
        tok_parts.push_back(pl.token_ids);
        for (int64_t i = 0; i < pl.num_tokens(); ++i) {
            rec_addr.push_back(pl.base_addr + i);
        }
    }
    at::Tensor rec_tok_c = at::cat(tok_parts).to(at::kCPU, at::kLong).contiguous();
    std::vector<int64_t> rec_tok(rec_tok_c.data_ptr<int64_t>(), rec_tok_c.data_ptr<int64_t>() + rec_tok_c.numel());
    const int64_t R = static_cast<int64_t>(rec_addr.size());
    // Per recalled ROW, the score of the page it came from, in the same payload order.
    std::vector<double> rec_score;
    rec_score.reserve(R);
    for (const Payload& pl : payloads) {
        rec_score.insert(rec_score.end(), static_cast<size_t>(pl.num_tokens()), pl.score);
    }

    // Recalled K/V per layer as [R, n_kv_heads, head_dim] on the pool device/dtype,
    // row r matching rec_addr[r]. Each payload's carrier is one contiguous
    // [cnt, n_layers, n_kv_heads, head_dim] block in the holding bucket, so a page
    // uploads as ONE transfer straight out of the bucket with nothing assembled on the
    // host; the device staging then transposes to the layer-major order the pool takes.
    const int64_t nkv = this->active->k_pool(0).size(2);
    const int64_t hd = this->active->k_pool(0).size(3);
    auto dev_opts = at::TensorOptions().dtype(this->active->k_pool(0).scalar_type()).device(this->device);
    at::Tensor stage_k = at::empty({R, n_layers, nkv, hd}, dev_opts);
    at::Tensor stage_v = at::empty({R, n_layers, nkv, hd}, dev_opts);
    int64_t stage_row = 0;
    for (const Payload& pl : payloads) {
        const int64_t cnt = pl.num_tokens();
        stage_k.narrow(0, stage_row, cnt).copy_(pl.k);
        stage_v.narrow(0, stage_row, cnt).copy_(pl.v);
        stage_row += cnt;
    }
    at::Tensor recalled_k = stage_k.transpose(0, 1).contiguous();
    at::Tensor recalled_v = stage_v.transpose(0, 1).contiguous();
    std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        rec_k[l] = recalled_k.select(0, l);
        rec_v[l] = recalled_v.select(0, l);
    }

    // Merge survivors and recalled, stable-sorted by addr -> temporal order. A survivor
    // carries its old SLOT INDEX (the page-alignment coordinate) plus the baked position
    // that index holds under the buffer's CURRENT layout; the two differ whenever that
    // layout is compacted. A recalled token has no slot index and its baked position is
    // its stored demotion pos, the session's short_offset.
    struct Entry {
        int64_t addr;
        int64_t old_pos;
        int64_t old_rope;
        int64_t tok;
        int64_t row;
        bool survivor;
    };
    std::vector<Entry> ent;
    ent.reserve(K + R);
    for (int64_t i = 0; i < K; ++i) {
        ent.push_back(Entry{surv_addr[i], i, this->active->rope_pos_at(seq_id, i), surv_tok[i], -1, true});
    }
    for (int64_t r = 0; r < R; ++r) {
        ent.push_back(Entry{rec_addr[r], -1, short_offset, rec_tok[r], r, false});
    }
    std::stable_sort(ent.begin(), ent.end(), [](const Entry& a, const Entry& b) { return a.addr < b.addr; });

    const int64_t N = static_cast<int64_t>(ent.size());
    RecompactPlan plan;
    plan.survivor.resize(N);
    plan.old_pos.resize(N);
    plan.old_rope.resize(N);
    plan.recalled_row.resize(N);
    plan.addr.resize(N);
    plan.tok.resize(N);
    plan.score.resize(N);
    for (int64_t i = 0; i < N; ++i) {
        plan.survivor[i] = ent[i].survivor ? 1 : 0;
        plan.old_pos[i] = ent[i].old_pos;
        plan.old_rope[i] = ent[i].old_rope;
        plan.recalled_row[i] = ent[i].row;
        plan.addr[i] = ent[i].addr;
        plan.tok[i] = ent[i].tok;
        plan.score[i] = ent[i].survivor ? 0.0 : rec_score[ent[i].row];
    }

    // recompact_with rebuilds the active page_id/tok to the merged order; the buckets
    // keep no such map.
    PULSAR_PROF_EXPR(
        RECOMPACT_GPU,
        this->active->recompact_with(
            seq_id,
            this->seqs.at(seq_id).n_sink,
            cfg.n_working,
            cfg.position_layout,
            short_offset,
            plan,
            rec_k,
            rec_v,
            this->seqs.at(seq_id).stream_len,
            cfg.attention_mass_decay
        )
    );
}

void Engine::run_cycle(int64_t seq_id, const RecallQuery& query, bool decode) {
    SeqState& st = this->seqs.at(seq_id);
    st.stats.cycles += 1;
    // Ahead of the demotion rotations and query positions, which read the layout.
    this->active->relayout(seq_id, st.n_sink, st.cfg.n_working, st.cfg.position_layout, st.short_offset);
    // Eviction runs in both phases: the bar is an absolute density, so residency settles
    // wherever pages stop clearing it rather than at active_buffer_size, which is only a
    // hard cap. Recall is what recall_during_prefill gates.
    this->evict_demote(seq_id, decode);  // gates internally
    if (decode) {
        ++st.stats.decode_cycles;
    }
    const int64_t after_evict = this->active->active_len(seq_id);
    st.stats.post_evict_tokens += after_evict;
    if (decode) {
        st.stats.post_evict_tokens_decode += after_evict;
    }
    if (this->active_max_size - after_evict <= 0) {
        ++st.stats.starved_cycles;
        if (decode) {
            ++st.stats.starved_cycles_decode;
        }
    }
    if (decode || st.cfg.recall_during_prefill) {
        this->recall_recompact(seq_id, query, decode);
    }
    const int64_t after_recall = this->active->active_len(seq_id);
    st.stats.post_recall_tokens += after_recall;
    if (decode) {
        st.stats.post_recall_tokens_decode += after_recall;
    }
    this->end_cycle(seq_id);
}

void Engine::recall_cycle(const BatchDescriptor& desc) {
    PULSAR_PROF_SCOPE(RECALL_CYCLE);
    std::set<int64_t> prefilling;
    const c10::List<int64_t>& pre = desc.prefill_seq_ids;
    if (!pre.empty()) {
        // These descriptors live on the device, so a step with no prefill group must not
        // pay their D2H.
        at::Tensor cu = desc.prefill_cu_seqlens_q.to(at::kCPU, at::kLong).contiguous();
        at::Tensor starts = desc.prefill_prompt_starts.to(at::kCPU, at::kLong).contiguous();
        at::Tensor seqlens_k = desc.prefill_seqlens_k.to(at::kCPU, at::kLong).contiguous();
        auto cu_a = cu.accessor<int64_t, 1>();
        auto starts_a = starts.accessor<int64_t, 1>();
        auto seqlens_k_a = seqlens_k.accessor<int64_t, 1>();
        for (size_t si = 0; si < pre.size(); ++si) {
            const int64_t sid = pre.get(si);
            prefilling.insert(sid);
            if (!this->sched->is_running(sid)) {
                continue;
            }
            const SessionRuntime& cfg = this->seqs.at(sid).cfg;
            const int64_t chunk = cu_a[si + 1] - cu_a[si];
            const int64_t prompt_start = starts_a[si];
            // The rerank query is the chunk's LAST query_window tokens, the same
            // window the decode pass reads off its ring. A window at or above
            // max_chunk_size takes the whole chunk.
            const int64_t query_len = std::min(cfg.query_window, chunk);
            const int64_t query_lo = cu_a[si + 1] - query_len;  // prefill-group row
            RecallQuery query;
            query.vecs = this->gather_query_q(query_lo, cu_a[si + 1]);
            query.lse = this->gather_query_lse(query_lo, cu_a[si + 1]);
            // A prefill row attends its CAUSAL prefix, so its key count is the context
            // position it sits at plus one. The chunk begins at context position
            // seqlens_k - chunk, NOT at its conversation address, which runs ahead once
            // eviction has removed anything.
            const int64_t ctx_start = seqlens_k_a[si] - chunk;
            std::vector<int64_t> attended(query_len);
            for (int64_t i = 0; i < query_len; ++i) {
                attended[i] = ctx_start + (chunk - query_len) + i + 1;
            }
            query.log_attended_len = log_attended(attended);
            // The BM25 terms are the last bm25_query_tokens addresses, independent of the
            // rerank window: one term ranks segments by a single word's tf-idf, which
            // leaves every segment lacking it tied at zero.
            const int64_t chunk_end = prompt_start + chunk;
            query.token_ids = this->token_ids_at(sid, bm25_query_start(chunk_end, cfg.bm25_query_tokens), chunk_end);
            this->run_cycle(sid, query, /*decode=*/false);
        }
    }
    for (auto& [sid, st] : this->seqs) {
        if (!this->sched->is_running(sid) || prefilling.count(sid)) {
            continue;
        }
        if (++st.decode_since_cycle < this->decode_cycle_interval) {
            continue;
        }
        st.decode_since_cycle = 0;
        RecallQuery query = this->read_q_ring(sid);
        // BM25 terms over the last bm25_query_tokens addresses (see the prefill pass); the
        // rerank window governs only the ring the Q comes from. The addresses below the
        // tail are cached; the just-sampled last_token (address stream_len - 1) has no KV
        // yet, so it is appended explicitly.
        const int64_t lo = bm25_query_start(st.stream_len, st.cfg.bm25_query_tokens);
        query.token_ids = this->token_ids_at(sid, lo, st.stream_len - 1);
        query.token_ids.push_back(st.last_token);
        this->run_cycle(sid, query, /*decode=*/true);
    }
}

}  // namespace pulsar
