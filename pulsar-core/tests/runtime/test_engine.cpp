#include "pulsar/runtime/engine.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

// torch's logging header defines a CHECK macro; drop it so Catch2's CHECK wins.
#undef CHECK
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

// create_session() resolves one session's knobs and freezes them on the sequence, so
// two sessions in the same engine evict and recall under different bars, and a
// malformed override fails at create_session() with nothing enqueued. Requires CUDA
// (the engine builds its pools and its model on the device).

namespace {

using pulsar::EngineConfig;
using pulsar::SessionConfig;

// The smallest Qwen3 the paged kernels take. The weights are random and the outputs are
// meaningless: every assertion here is about which pages the cycle moves, not about
// what the model says.
constexpr int64_t N_LAYERS = 2;
constexpr int64_t N_HEADS = 4;
constexpr int64_t N_KV_HEADS = 2;
constexpr int64_t HEAD_DIM = 32;
constexpr int64_t HIDDEN = 128;
constexpr int64_t INTERMEDIATE = 256;
constexpr int64_t VOCAB = 200;
constexpr int64_t PAGE_SIZE = 16;
// 6 pages per session; the derived active_max_size is this minus one chunk, so 80 tokens.
constexpr int64_t ACTIVE_BUFFER = 6 * PAGE_SIZE;
constexpr int64_t MAX_CHUNK = 16;
constexpr double THETA = 1'000'000.0;

void require_cuda() {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
}

at::Tensor small(std::initializer_list<int64_t> shape) {
    return at::randn(shape, at::TensorOptions().dtype(at::kFloat)).mul_(0.05).to(at::kCUDA);
}

at::Tensor unit(int64_t n) {
    return at::ones({n}, at::TensorOptions().dtype(at::kFloat)).to(at::kCUDA);
}

c10::Dict<std::string, at::Tensor> tiny_weights() {
    at::manual_seed(0);
    c10::Dict<std::string, at::Tensor> w;
    const int64_t q_dim = N_HEADS * HEAD_DIM, kv_dim = N_KV_HEADS * HEAD_DIM;
    w.insert("model.embed_tokens.weight", small({VOCAB, HIDDEN}));
    w.insert("model.norm.weight", unit(HIDDEN));
    for (int64_t i = 0; i < N_LAYERS; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        w.insert(p + "input_layernorm.weight", unit(HIDDEN));
        w.insert(p + "post_attention_layernorm.weight", unit(HIDDEN));
        w.insert(p + "self_attn.q_proj.weight", small({q_dim, HIDDEN}));
        w.insert(p + "self_attn.k_proj.weight", small({kv_dim, HIDDEN}));
        w.insert(p + "self_attn.v_proj.weight", small({kv_dim, HIDDEN}));
        w.insert(p + "self_attn.o_proj.weight", small({HIDDEN, q_dim}));
        w.insert(p + "self_attn.q_norm.weight", unit(HEAD_DIM));
        w.insert(p + "self_attn.k_norm.weight", unit(HEAD_DIM));
        w.insert(p + "mlp.gate_proj.weight", small({INTERMEDIATE, HIDDEN}));
        w.insert(p + "mlp.up_proj.weight", small({INTERMEDIATE, HIDDEN}));
        w.insert(p + "mlp.down_proj.weight", small({HIDDEN, INTERMEDIATE}));
    }
    return w;
}

// Recall on (recall_buffer_size > 0) with both bars at their inert defaults, so a
// session that names no override evicts nothing.
EngineConfig recall_config() {
    EngineConfig c;
    c.model = "qwen3";
    c.weights = tiny_weights();
    c.n_layers = N_LAYERS;
    c.n_heads = N_HEADS;
    c.n_kv_heads = N_KV_HEADS;
    c.head_dim = HEAD_DIM;
    c.hidden = HIDDEN;
    c.intermediate = INTERMEDIATE;
    c.vocab = VOCAB;
    c.rope_theta = THETA;
    c.rms_eps = 1e-6;
    c.max_sessions = 2;
    c.active_buffer_size = ACTIVE_BUFFER;
    c.page_size = PAGE_SIZE;
    c.dtype = "float32";
    c.device = "cuda";
    c.max_running = 2;
    c.max_chunk_size = MAX_CHUNK;
    c.session.n_working = PAGE_SIZE;
    c.recall_buffer_size = 4096;
    return c;
}

c10::List<int64_t> ids(int64_t n, int64_t base) {
    c10::List<int64_t> out;
    for (int64_t i = 0; i < n; ++i) {
        out.push_back((base + i) % VOCAB);
    }
    return out;
}

// The decode-cadence cycles one session runs over n_decode decode steps: the run's
// cycle count taken across the prefill boundary, so the per-chunk cycles are subtracted
// out. The first step() drives the whole prefill and emits one token, so every later
// step is a decode step and every cycle after that baseline is a decode-cadence one.
int64_t decode_cycles(pulsar::Engine& engine, int64_t n_decode) {
    const int64_t seq = engine.create_session(ids(PAGE_SIZE, 1));
    engine.feed(seq, ids(64, 17));
    engine.step(1);
    const int64_t after_prefill = engine.stats(seq).cycles;
    engine.step(n_decode);
    return engine.stats(seq).cycles - after_prefill;
}

}  // namespace

// Two sessions in ONE engine, differing only in evict_tau. The evict->recall cycle runs
// per sequence after the batched forward and reads each one's own resolved bar, so the
// session at tau ~ 1 (a bar every real page is under) demotes and refills while the
// session at the default tau 0 (a bar nothing is under) leaves its pages alone.
TEST_CASE("two sessions evict and recall under their own bars", "[engine][cuda]") {
    require_cuda();
    pulsar::Engine engine(recall_config());

    SessionConfig evicting = engine.session_defaults();
    evicting.evict_tau = 0.9999;  // bar 9999x uniform: no real page clears it
    // This exercise never leaves prefill, so both sessions opt into prefill recall and
    // evict_tau stays the only difference between them.
    evicting.recall_during_prefill = true;
    SessionConfig intact_cfg = engine.session_defaults();
    intact_cfg.recall_during_prefill = true;

    const int64_t seq0 = engine.create_session(ids(PAGE_SIZE, 1), evicting);
    const int64_t seq1 = engine.create_session(ids(PAGE_SIZE, 1), intact_cfg);  // tau 0
    // Content up to active_max_size (80 tokens), so its backstop never fires and
    // eviction is the bar's decision alone.
    engine.feed(seq0, ids(64, 17));
    engine.feed(seq1, ids(64, 17));
    // One emission per sequence lands only once its whole prompt is prefilled, and the
    // prefill budget is shared, so four tokens takes both through their chunk cycles.
    engine.step(4);

    const pulsar::Stats evicted = engine.stats(seq0);
    const pulsar::Stats intact = engine.stats(seq1);
    CHECK(evicted.cycles > 0);
    CHECK(intact.cycles > 0);
    CHECK(evicted.evicted_pages > 0);
    CHECK(evicted.recalled_pages > 0);  // its recall bar is still 0, so it refills
    CHECK(intact.evicted_pages == 0);
    CHECK(intact.recalled_pages == 0);
}

// A malformed value fails at create_session(), before anything is enqueued: the
// session's knobs are resolved once, so a bad one can never surface mid-run.
TEST_CASE("a malformed session override throws at create_session", "[engine][cuda]") {
    require_cuda();
    pulsar::Engine engine(recall_config());

    SessionConfig partial_page = engine.session_defaults();
    partial_page.block_radius = PAGE_SIZE / 2;  // not a whole number of pages
    CHECK_THROWS(engine.create_session(ids(PAGE_SIZE, 1), partial_page));

    SessionConfig no_query = engine.session_defaults();
    no_query.recall_query_window = 0;  // the rerank needs at least one query token
    CHECK_THROWS(engine.create_session(ids(PAGE_SIZE, 1), no_query));

    SessionConfig split_segment = engine.session_defaults();
    split_segment.recall_segment_size = PAGE_SIZE + 1;  // not a multiple of the page
    CHECK_THROWS(engine.create_session(ids(PAGE_SIZE, 1), split_segment));

    // None of the failed creates opened a session.
    CHECK(engine.active_sequence_count() == 0);
    engine.create_session(ids(PAGE_SIZE, 1));
    CHECK(engine.active_sequence_count() == 1);

    // alpha == 1 (retention 0, a one-token memory) is the top of the range.
    SessionConfig one_token_memory = engine.session_defaults();
    one_token_memory.attention_mass_decay = 1.0;
    engine.create_session(ids(PAGE_SIZE, 1), one_token_memory);
    CHECK(engine.active_sequence_count() == 2);
}

// n_working has no default, so a recalling session that leaves it unset is a
// configuration error rather than a silent fallback to max_chunk_size. With recall off
// it is unread, and an engine that never evicts must still open.
TEST_CASE("n_working must be stated when recall is on", "[engine][cuda]") {
    require_cuda();
    pulsar::Engine engine(recall_config());
    SessionConfig unset = engine.session_defaults();
    unset.n_working = 0;
    CHECK_THROWS(engine.create_session(ids(PAGE_SIZE, 1), unset));
    CHECK(engine.active_sequence_count() == 0);

    EngineConfig off = recall_config();
    off.recall_buffer_size = 0;
    off.session.n_working = 0;
    pulsar::Engine no_recall(off);
    no_recall.create_session(ids(PAGE_SIZE, 1));
    CHECK(no_recall.active_sequence_count() == 1);
}

// A buffer smaller than one chunk leaves no room to cycle in, so active_max_size is 0.
// That is legal with recall OFF: the buffer is sized to the whole prompt and
// max_chunk_size is a bound nothing reaches, so only a session that will cycle has to
// satisfy it.
TEST_CASE("a buffer under one chunk opens a session when recall is off", "[engine][cuda]") {
    require_cuda();
    EngineConfig cfg = recall_config();
    cfg.active_buffer_size = 4 * PAGE_SIZE;
    cfg.max_chunk_size = 8 * PAGE_SIZE;  // larger than the whole buffer
    cfg.recall_buffer_size = 0;  // recall off: nothing evicts or cycles
    pulsar::Engine engine(cfg);
    engine.create_session(ids(PAGE_SIZE, 1));
    CHECK(engine.active_sequence_count() == 1);
}

// alpha resolves against active_max_size. 0 (and anything under the floor) derives
// 1 / active_max_size; the floor is also the slowest rate allowed, since a slower one
// would remember past the window the bias correction has age to divide out. Above 1 caps
// at 1.
TEST_CASE("attention_mass_decay derives and clamps against active_max_size", "[engine]") {
    CHECK(pulsar::resolve_mass_decay(0.0, 4096) == Catch::Approx(1.0 / 4096));
    CHECK(pulsar::resolve_mass_decay(0.0, 1024) == Catch::Approx(1.0 / 1024));
    // Explicit and inside the range: used as given.
    CHECK(pulsar::resolve_mass_decay(0.01, 4096) == Catch::Approx(0.01));
    // Under the floor for this active_max_size, so it is raised to it.
    CHECK(pulsar::resolve_mass_decay(1.0 / 8192, 4096) == Catch::Approx(1.0 / 4096));
    // The same rate is left alone under an active_max_size that can measure it.
    CHECK(pulsar::resolve_mass_decay(1.0 / 8192, 16384) == Catch::Approx(1.0 / 8192));
    CHECK(pulsar::resolve_mass_decay(1.5, 4096) == Catch::Approx(1.0));
    CHECK(pulsar::resolve_mass_decay(-1.0, 4096) == Catch::Approx(1.0 / 4096));
    // Recall off: no window to measure against and nothing accumulates.
    CHECK(pulsar::resolve_mass_decay(0.5, 0) == 0.0);
}

// <= 0 takes max_chunk_size, a positive value is taken as given, and an interval that
// resolves to 0 (no chunk budget to fall back to) is rejected.
TEST_CASE("decode_cycle_interval resolves against max_chunk_size", "[engine]") {
    CHECK(pulsar::resolve_decode_cycle_interval(0, 512) == 512);
    CHECK(pulsar::resolve_decode_cycle_interval(-1, 512) == 512);
    CHECK(pulsar::resolve_decode_cycle_interval(64, 512) == 64);
    CHECK(pulsar::resolve_decode_cycle_interval(1, 512) == 1);
    CHECK_THROWS(pulsar::resolve_decode_cycle_interval(0, 0));
    CHECK_THROWS(pulsar::resolve_decode_cycle_interval(-1, -8));
}

// <= 0 takes n_working, so an unset knob keeps the whole protected window as the BM25
// query; a positive value is taken as given. A query that resolves to 0 tokens has no
// terms and is rejected.
TEST_CASE("recall_bm25_query_tokens resolves against n_working", "[engine]") {
    CHECK(pulsar::resolve_bm25_query_tokens(0, 8192) == 8192);
    CHECK(pulsar::resolve_bm25_query_tokens(-1, 8192) == 8192);
    CHECK(pulsar::resolve_bm25_query_tokens(256, 8192) == 256);
    CHECK(pulsar::resolve_bm25_query_tokens(1, 8192) == 1);
    // Above the protected window the query simply reaches further back in the stream.
    CHECK(pulsar::resolve_bm25_query_tokens(16384, 8192) == 16384);
    CHECK_THROWS(pulsar::resolve_bm25_query_tokens(0, 0));
    CHECK_THROWS(pulsar::resolve_bm25_query_tokens(-4, 0));
}

// The query the cycle hands BM25 is exactly the resolved token count, off the end of the
// stream, and is truncated only where the stream is shorter than that.
TEST_CASE("the bm25 query is the resolved token count long", "[engine]") {
    constexpr int64_t END = 10'000;
    for (int64_t tokens : {1, 256, 8192}) {
        const int64_t start = pulsar::bm25_query_start(END, tokens);
        CHECK(END - start == tokens);
    }
    CHECK(pulsar::bm25_query_start(100, 256) == 0);  // clamped at the stream head
    CHECK(pulsar::bm25_query_start(0, 256) == 0);
}

// The decode cadence is its own knob: an engine that leaves it unset cycles once per
// max_chunk_size generated tokens, and a configured interval replaces that count.
TEST_CASE("decode_cycle_interval sets the decode cadence", "[engine][cuda]") {
    require_cuda();
    constexpr int64_t N_DECODE = 4 * MAX_CHUNK;

    pulsar::Engine unset(recall_config());
    CHECK(decode_cycles(unset, N_DECODE) == N_DECODE / MAX_CHUNK);

    EngineConfig restated = recall_config();
    restated.decode_cycle_interval = MAX_CHUNK;
    pulsar::Engine same_cadence(restated);
    CHECK(decode_cycles(same_cadence, N_DECODE) == N_DECODE / MAX_CHUNK);

    EngineConfig halved = recall_config();
    halved.decode_cycle_interval = MAX_CHUNK / 2;
    pulsar::Engine tighter(halved);
    CHECK(decode_cycles(tighter, N_DECODE) == N_DECODE / (MAX_CHUNK / 2));
}
