#include "pulsar/runtime/kv/active_buffer.hpp"
#include "pulsar/runtime/engine.hpp"
#include "pulsar/ops.hpp"
#include "pulsar/rope.hpp"
#include "pulsar/runtime/scheduler.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

// torch's logging header defines a CHECK macro; drop it so Catch2's CHECK wins.
#undef CHECK
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Covers the ActiveBuffer + continuous-batching Scheduler: allocate/append/free
// accounting, OOM, slot_mapping, write_kv round-trip, page-level density eviction
// with the sink/working band (absolute [0, 1] keep-bar, bias-corrected max-pooled
// density, age correction, survivor in-place re-rope + DemotedKV contents,
// active_cap backstop), decay_mass, the EMA upkeep leaky integrator, recall
// recompaction + seed-to-bar, and the scheduler admission / chunked prefill /
// mixed step behavior. Requires CUDA.

namespace {

using Catch::Approx;
using pulsar::ActiveBuffer;
using pulsar::DemotedKV;
using pulsar::Scheduler;
using enum pulsar::PositionLayout;  // contiguous, compacted

const at::Device DEV(at::kCUDA);
constexpr double THETA = 1'000'000.0;
// The absolute position demotion ropes victim keys to. It sits above every slot index
// these tests use, so a demoted key is never already at the position it is rotated to
// and the rotation cannot pass by doing nothing.
constexpr int64_t SHORT_OFFSET = 4096;

void require_cuda() {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
}

ActiveBuffer make_alloc(
    int64_t num_pages = 32,
    int64_t page_size = 16,
    int64_t n_kv_heads = 2,
    int64_t head_dim = 64,
    int64_t n_layers = 1,
    int64_t n_q_heads = 0,
    std::string dtype = "float32"
) {
    if (n_q_heads <= 0) {
        n_q_heads = n_kv_heads;  // group 1 unless overridden
    }
    return ActiveBuffer(
        n_layers,
        num_pages,
        page_size,
        n_kv_heads,
        n_q_heads,
        head_dim,
        std::move(dtype),
        "cuda",
        THETA
    );
}

// RoPE a [rows, n_kv_heads, head_dim] tensor at the given per-row pos.
at::Tensor rope_rows(const at::Tensor& x, const at::Tensor& pos) {
    auto roped = pulsar::rope_cuda(x.transpose(0, 1).contiguous(), pos, THETA);
    return roped.transpose(0, 1).contiguous();
}

std::vector<int64_t> list_to_vec(const c10::List<int64_t>& l) {
    std::vector<int64_t> v;
    for (size_t i = 0; i < l.size(); ++i) {
        v.push_back(l.get(i));
    }
    return v;
}
std::vector<int64_t> cpu_i32_vec(const at::Tensor& t) {
    auto c = t.to(at::kCPU, at::kLong).contiguous();
    auto a = c.accessor<int64_t, 1>();
    std::vector<int64_t> v;
    for (int64_t i = 0; i < c.numel(); ++i) {
        v.push_back(a[i]);
    }
    return v;
}

}  // namespace

TEST_CASE("ActiveBuffer allocate and free accounting", "[allocator][cuda]") {
    require_cuda();
    auto a = make_alloc(10, 16);
    CHECK(a.num_free_pages() == 10);
    a.allocate(0, 20);
    CHECK(a.num_free_pages() == 8);
    CHECK(a.active_len(0) == 20);
    CHECK(a.page_table(0).numel() == 2);
    a.allocate(1, 16);
    CHECK(a.num_free_pages() == 7);
    a.free(0);
    CHECK(a.num_free_pages() == 9);
    a.free(1);
    CHECK(a.num_free_pages() == 10);
}

TEST_CASE("ActiveBuffer append across page boundary", "[allocator][cuda]") {
    require_cuda();
    auto a = make_alloc(10, 16);
    a.allocate(0, 10);
    CHECK(a.page_table(0).numel() == 1);
    CHECK(a.pages_to_append(0, 5) == 0);
    a.append(0, 5);
    CHECK(a.active_len(0) == 15);
    CHECK(a.pages_to_append(0, 3) == 1);
    a.append(0, 3);
    CHECK(a.active_len(0) == 18);
    CHECK(a.page_table(0).numel() == 2);
    CHECK(a.num_free_pages() == 8);
}

TEST_CASE("ActiveBuffer OOM path", "[allocator][cuda]") {
    require_cuda();
    auto a = make_alloc(4, 16);
    CHECK(a.can_allocate(64));
    CHECK_FALSE(a.can_allocate(65));
    a.allocate(0, 64);
    CHECK(a.num_free_pages() == 0);
    CHECK_THROWS_AS(a.allocate(1, 1), std::exception);
}

TEST_CASE("ActiveBuffer slot_mapping matches page table", "[allocator][cuda]") {
    require_cuda();
    auto a = make_alloc(16, 16);
    a.allocate(0, 40);
    auto bt = a.page_table(0).to(at::kCPU);
    auto bta = bt.accessor<int32_t, 1>();
    const int bs = 16;

    SECTION("full window") {
        auto sm = a.slot_mapping(0, 0, 40).to(at::kCPU);
        auto sma = sm.accessor<int32_t, 1>();
        bool ok = true;
        for (int pos = 0; pos < 40; ++pos) {
            ok = ok && (sma[pos] == bta[pos / bs] * bs + (pos % bs));
        }
        CHECK(ok);
    }

    SECTION("offset window") {
        auto sm2 = a.slot_mapping(0, 17, 10).to(at::kCPU);
        auto sm2a = sm2.accessor<int32_t, 1>();
        bool ok = true;
        for (int i = 0; i < 10; ++i) {
            int pos = 17 + i;
            ok = ok && (sm2a[i] == bta[pos / bs] * bs + (pos % bs));
        }
        CHECK(ok);
    }
}

TEST_CASE("ActiveBuffer write_kv round-trip", "[allocator][cuda]") {
    require_cuda();
    auto a = make_alloc(16, 16, 2, 64);
    a.allocate(0, 20);
    auto k_pool = a.k_pool(0), v_pool = a.v_pool(0);
    auto slots = a.slot_mapping(0, 0, 20);
    auto k_new = at::randn({20, 2, 64}, at::device(DEV));
    auto v_new = at::randn({20, 2, 64}, at::device(DEV));
    pulsar::write_kv_cuda(k_pool, v_pool, k_new, v_new, slots);
    auto flat_k = k_pool.view({-1, 2, 64}).index_select(0, slots.to(at::kLong));
    auto flat_v = v_pool.view({-1, 2, 64}).index_select(0, slots.to(at::kLong));
    CHECK(at::equal(flat_k, k_new));
    CHECK(at::equal(flat_v, v_new));
}

TEST_CASE("decay_mass scales active mass", "[eviction][cuda]") {
    require_cuda();
    at::manual_seed(2);
    const int64_t bs = 16, n_kv = 2, hd = 64, nl = 2, ctx = 20;
    auto a = make_alloc(16, bs, n_kv, hd, nl);
    a.allocate(0, ctx);
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    for (int64_t l = 0; l < nl; ++l) {
        a.mass_pool(l).view({-1, n_kv}).index_copy_(0, slots, at::rand({ctx, n_kv}, at::device(DEV)) + 1.0);
    }

    std::vector<at::Tensor> before;
    for (int64_t l = 0; l < nl; ++l) {
        before.push_back(a.mass_pool(l).clone());
    }

    SECTION("factor 1.0 is a no-op") {
        a.decay_mass(0, 1.0);
        for (int64_t l = 0; l < nl; ++l) {
            CHECK(at::equal(a.mass_pool(l), before[l]));
        }
    }

    SECTION("factor 0.5 halves") {
        a.decay_mass(0, 0.5);
        for (int64_t l = 0; l < nl; ++l) {
            auto got = a.mass_pool(l).view({-1, n_kv}).index_select(0, slots);
            auto exp = before[l].view({-1, n_kv}).index_select(0, slots) * 0.5;
            CHECK(at::allclose(got, exp));
        }
    }
}

// upkeep() is the per-forward EMA / leaky-integrator step the Engine runs: it
// decays a sequence's active mass in place BEFORE the forward adds this step's
// received mass, so mass evolves as m_new = retention*m_old + received, retention =
// 1 - decay (decay is alpha, the forgetting rate). Drive the loop by hand (upkeep
// then a manual "received" write) and assert the two closed-form limits: a token
// attended once then ignored fades as received*retention^t, and a token attended
// every step converges to the fixed point received/(1-retention).
TEST_CASE("upkeep runs the attention-mass EMA leaky integrator", "[eviction][cuda]") {
    require_cuda();
    const int64_t bs = 16, n_kv = 1, hd = 64, nl = 1, ctx = 4;
    const double decay = 0.5, retention = 1.0 - decay, received = 3.0;
    const int64_t steps = 20;
    auto a = make_alloc(16, bs, n_kv, hd, nl);
    a.allocate(0, ctx);

    auto mass = a.mass_pool(0).view({-1, n_kv});  // [num_pages*page_size, n_kv]
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    const int64_t s_fade = slots[0].item<int64_t>();  // attended once, then ignored
    const int64_t s_steady = slots[1].item<int64_t>();  // attended every step

    // Step 0: no old mass to decay yet; the forward writes received to both keys.
    mass.index_put_({s_fade, 0}, received);
    mass.index_put_({s_steady, 0}, received);

    // Steps 1..N: decay OLD mass (upkeep), then the "forward" adds received only to
    // the steadily-attended key; the fade key gets nothing.
    for (int64_t t = 1; t <= steps; ++t) {
        // One token per step -> retention^1 (the per-token EMA).
        a.upkeep(0, 1, decay);
        double cur = mass.index({s_steady, 0}).item<double>();
        mass.index_put_({s_steady, 0}, cur + received);
    }

    double fade = mass.index({s_fade, 0}).item<double>();
    double steady = mass.index({s_steady, 0}).item<double>();

    // Fade: one deposit at step 0 decayed `steps` times -> received * retention^steps.
    // The horizon the steady key needs to converge leaves the faded mass at 2.9e-6, so
    // the bound has to be RELATIVE: an absolute one would be satisfied by zero.
    const double expected_fade = received * std::pow(retention, steps);
    CHECK(std::abs(fade - expected_fade) < 1e-4 * expected_fade);
    // Steady: geometric series converges to the leaky-integrator fixed point.
    CHECK(std::abs(steady - received / (1.0 - retention)) < 1e-3);
}

// A recall's recompact must lay the active survivors together with the recalled
// tokens out in address (temporal) order and re-rope every key to its new
// contiguous pos, so a active key equals rope(raw_key, addr_rank): a token
// with addr a lands at the pos that its addr holds in the merged order, not at
// the growing tail in relevance order. It must hold across a SECOND evict->recall
// cycle: the second cycle re-ropes correctly only if the first left every token's
// logical index equal to its baked rotation. The recalled KV is taken straight
// from evict()'s DemotedKV (exactly
// what the RecallBuffer round-trips), so this drives the real evict + recompact_with
// data flow without a model. Multiple demotes before one recall give a recalled
// token whose demotion pos differs from its final addr rank (a non-zero
// recalled re-rope), and survivors that shift across the merge (a non-zero survivor
// re-rope).
TEST_CASE("recall recompacts survivors and recalled into temporal addr order", "[recall][recompact][cuda]") {
    require_cuda();
    at::manual_seed(0);
    const int64_t bs = 16, n_kv = 2, hd = 64, n_layers = 2, ctx = 64;  // 4 pages
    auto a = make_alloc(64, bs, n_kv, hd, n_layers);
    a.allocate(0, ctx);

    // Raw (un-roped) K/V per layer; write them roped at their initial pos.
    // addr[p] tracks the address of the token at current logical pos p;
    // for a fresh prefill addr[p] == p, and raw_k[l][addr] is that token's raw key.
    auto pos_full = at::arange(ctx, at::TensorOptions().dtype(at::kLong).device(DEV));
    auto slots = a.slot_mapping(0, 0, ctx);
    std::vector<at::Tensor> raw_k(n_layers), raw_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        raw_k[l] = at::randn({ctx, n_kv, hd}, at::device(DEV));
        raw_v[l] = at::randn({ctx, n_kv, hd}, at::device(DEV));
        pulsar::write_kv_cuda(a.k_pool(l), a.v_pool(l), rope_rows(raw_k[l], pos_full), raw_v[l], slots);
    }
    std::vector<int64_t> addr(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
    }

    auto to_long_dev = [&](const std::vector<int64_t>& v) {
        return at::tensor(v, at::TensorOptions().dtype(at::kLong)).to(DEV);
    };

    // Every active logical pos p must hold rope(raw_k[addr[p]], p), and V
    // must be the untouched raw_v[addr[p]].
    auto check_active = [&]() {
        const int64_t n = a.active_len(0);
        REQUIRE(n == static_cast<int64_t>(addr.size()));
        auto new_slots = a.slot_mapping(0, 0, n).to(at::kLong);
        auto new_pos = at::arange(n, at::TensorOptions().dtype(at::kLong).device(DEV));
        auto addr_idx = to_long_dev(addr);
        for (int64_t l = 0; l < n_layers; ++l) {
            auto got_k = a.k_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
            auto exp_k = rope_rows(raw_k[l].index_select(0, addr_idx), new_pos);
            CHECK(at::allclose(got_k, exp_k, 1e-3, 1e-3));
            auto got_v = a.v_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
            auto exp_v = raw_v[l].index_select(0, addr_idx);
            CHECK(at::equal(got_v, exp_v));
        }
    };

    // Set per-slot mass over the current layout so the given LOGICAL pages are the
    // lowest-mass eviction candidates.
    auto set_page_mass = [&](const std::vector<int64_t>& low_pages) {
        const int64_t n = a.active_len(0);
        auto cur = a.slot_mapping(0, 0, n).to(at::kLong);
        auto mass = at::full({n}, 100.0, at::device(DEV).dtype(at::kFloat));
        for (int64_t lp : low_pages) {
            const int64_t lo = lp * bs, hi = std::min((lp + 1) * bs, n);
            for (int64_t p = lo; p < hi; ++p) {
                mass[p] = 1.0;
            }
        }
        for (int64_t l = 0; l < n_layers; ++l) {
            auto mflat = a.mass_pool(l).view({-1, n_kv});
            for (int64_t h = 0; h < n_kv; ++h) {
                mflat.index_put_({cur, h}, mass);
            }
        }
    };

    // Pending demoted set accumulated across evicts, recalled in one recompact.
    std::vector<int64_t> pend_addr, pend_pos;
    std::vector<std::vector<at::Tensor>> pend_k(n_layers), pend_v(n_layers);

    auto evict_only = [&](const std::vector<int64_t>& low_pages, int64_t target) {
        set_page_mass(low_pages);
        const int64_t before = a.active_len(0);
        // Bar off (min_density 0); the active_cap backstop demotes the
        // ceil((ctx - target) / page_size) lowest-density pages (the low_pages).
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/target,
            /*block_radius=*/0,
            /*stream_len=*/before
        );
        // Demoted K is roped at SHORT_OFFSET. victim logical pos come from the
        // low-mass whole pages (all evicted: their count equals the pages the eviction
        // drops), gathered ascending to match the demoted victim-slot row order.
        std::vector<int64_t> dpos;
        for (int64_t lp : low_pages) {
            const int64_t lo = lp * bs, hi = std::min((lp + 1) * bs, before);
            for (int64_t p = lo; p < hi; ++p) {
                dpos.push_back(p);
            }
        }
        std::sort(dpos.begin(), dpos.end());
        std::unordered_set<int64_t> victim(dpos.begin(), dpos.end());
        for (int64_t p : dpos) {
            pend_pos.push_back(SHORT_OFFSET);  // the demotion position K carries
            pend_addr.push_back(addr[p]);  // demoted token's addr identity
        }
        // Gather the demoted K/V from the freed-but-live victim slots, staged
        // [n_kv, M, hd] so later evict_only calls' rows concat along dim=1;
        // recall_pending transposes back to [M, n_kv, hd] before recompact_with.
        for (int64_t l = 0; l < n_layers; ++l) {
            pend_k[l].push_back(
                a.k_pool(l).view({-1, n_kv, hd}).index_select(0, d.victim_slots).transpose(0, 1).contiguous()
            );
            pend_v[l].push_back(
                a.v_pool(l).view({-1, n_kv, hd}).index_select(0, d.victim_slots).transpose(0, 1).contiguous()
            );
        }
        std::vector<int64_t> surv;  // survivor addr, in compacted logical order
        for (size_t p = 0; p < addr.size(); ++p) {
            if (!victim.count(static_cast<int64_t>(p))) {
                surv.push_back(addr[p]);
            }
        }
        addr = std::move(surv);
        REQUIRE(static_cast<int64_t>(addr.size()) == a.active_len(0));
    };

    auto recall_pending = [&]() {
        const int64_t K = a.active_len(0);
        const int64_t R = static_cast<int64_t>(pend_addr.size());
        std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers);
        for (int64_t l = 0; l < n_layers; ++l) {
            rec_k[l] = at::cat(pend_k[l], /*dim=*/1).transpose(0, 1).contiguous().to(DEV);
            rec_v[l] = at::cat(pend_v[l], /*dim=*/1).transpose(0, 1).contiguous().to(DEV);
        }
        // A survivor's slot index is its baked pos under this contiguous layout; a
        // recalled token has no slot index and carries the demotion position instead.
        struct E {
            int64_t addr, old_pos, old_rope, row;
            bool surv;
        };
        std::vector<E> ent;
        for (int64_t i = 0; i < K; ++i) {
            ent.push_back({addr[i], i, i, -1, true});
        }
        for (int64_t r = 0; r < R; ++r) {
            ent.push_back({pend_addr[r], -1, pend_pos[r], r, false});
        }
        std::stable_sort(ent.begin(), ent.end(), [](const E& x, const E& y) { return x.addr < y.addr; });
        const int64_t N = static_cast<int64_t>(ent.size());
        pulsar::RecompactPlan plan;
        plan.survivor.resize(N);
        plan.old_pos.resize(N);
        plan.old_rope.resize(N);
        plan.recalled_row.resize(N);
        plan.addr.resize(N);
        plan.tok.resize(N);
        plan.score.assign(N, 0.0);
        std::vector<int64_t> merged(N);
        for (int64_t i = 0; i < N; ++i) {
            plan.survivor[i] = ent[i].surv ? 1 : 0;
            plan.old_pos[i] = ent[i].old_pos;
            plan.old_rope[i] = ent[i].old_rope;
            plan.recalled_row[i] = ent[i].row;
            plan.addr[i] = ent[i].addr;  // merged temporal order is contiguous [0, N)
            plan.tok[i] = ent[i].addr;  // token id stand-in (addr), unconstrained here
            merged[i] = ent[i].addr;
        }
        a.recompact_with(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            plan,
            rec_k,
            rec_v
        );
        addr = std::move(merged);
        pend_addr.clear();
        pend_pos.clear();
        for (int64_t l = 0; l < n_layers; ++l) {
            pend_k[l].clear();
            pend_v[l].clear();
        }
    };

    check_active();  // baseline: fresh prefill at [0, 64)

    // Cycle 1: two demotes before one recall. Demote the middle pages (addr 16..47),
    // then demote logical page 1 (now addr 48..63), then recall all. Merged temporal
    // order restores addr 0..63; recalled K is roped at SHORT_OFFSET, so addr 48..63
    // re-ropes SHORT_OFFSET -> new 48..63 and addr 16..47 lands at 16..47 (not the tail).
    evict_only({1, 2}, /*target=*/32);
    evict_only({1}, /*target=*/16);
    recall_pending();
    check_active();

    // Cycle 2: evict->recall again on the restored full sequence. Evicting the
    // front and back pages shifts the survivors (addr 16..47) across the merge
    // (survivor re-rope != 0); it only re-ropes correctly if cycle 1 left every
    // token's logical index == its baked rotation.
    evict_only({0, 3}, /*target=*/32);
    recall_pending();
    check_active();
}

// Recall is headroom-bounded. The engine gives recall the free tokens
// (max_context - active) as its capacity and selection stops itself on them, so
// active + recalled tokens <= max_context. This drives recompact_with with a
// page-aligned plan sized to as many whole pages as that capacity holds and checks the
// resulting active_len is exactly active + those pages' tokens and within the
// window. It also proves NO survivor K/V is copied: every physical survivor page is
// still in the permuted table (recalled pages are freshly claimed), survivor V is
// byte-identical (V is never touched), and each key equals rope(raw, addr_rank).
TEST_CASE("recall headroom bound caps segments and copies no survivor K/V", "[recall][recompact][headroom][cuda]") {
    require_cuda();
    at::manual_seed(0);
    const int64_t bs = 16, n_kv = 2, hd = 64, n_layers = 2;
    const int64_t segment_size = bs;  // one page per segment
    const int64_t max_context = 48;  // active window (3 pages)
    auto a = make_alloc(64, bs, n_kv, hd, n_layers);

    // Survivors: two full pages of recent tokens (addr 100..131) active at 0..31.
    const int64_t active = 32;
    a.allocate(0, active);
    const std::vector<int32_t> surv_pages_before = a.pages_of(0);
    auto pos_s = at::arange(active, at::TensorOptions().dtype(at::kLong).device(DEV));
    auto slots_s = a.slot_mapping(0, 0, active);
    std::vector<at::Tensor> surv_raw_k(n_layers), surv_raw_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        surv_raw_k[l] = at::randn({active, n_kv, hd}, at::device(DEV));
        surv_raw_v[l] = at::randn({active, n_kv, hd}, at::device(DEV));
        pulsar::write_kv_cuda(a.k_pool(l), a.v_pool(l), rope_rows(surv_raw_k[l], pos_s), surv_raw_v[l], slots_s);
    }
    // Survivor addresses are page-aligned whole pages (pages 3,4) with a gap after
    // the recalled page 0 (addr 0..15), the address-aligned invariant the engine
    // holds. addr 48..79 => the merged plan has aligned page bases {0, 48, 64}.
    std::vector<int64_t> surv_addr(active);
    for (int64_t i = 0; i < active; ++i) {
        surv_addr[i] = 48 + i;
    }

    // Headroom fill: how many whole pages the free tokens hold.
    const int64_t pages_that_fit = (max_context - active) / segment_size;
    REQUIRE(pages_that_fit == 1);
    const int64_t R = pages_that_fit * segment_size;

    // Recalled segment: R older tokens (addr 0..R-1), demoted (roped) at 0..R-1.
    auto rec_pos = at::arange(R, at::TensorOptions().dtype(at::kLong).device(DEV));
    std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers), rec_raw_k(n_layers), rec_raw_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        rec_raw_k[l] = at::randn({R, n_kv, hd}, at::device(DEV));
        rec_raw_v[l] = at::randn({R, n_kv, hd}, at::device(DEV));
        rec_k[l] = rope_rows(rec_raw_k[l], rec_pos);  // [R, n_kv, hd] roped at dpos
        rec_v[l] = rec_raw_v[l];
    }

    // Merge by addr (recalled addr 0..R-1 sort before survivor addr 100..) and build
    // the page-aligned RecompactPlan the engine hands recompact_with.
    struct E {
        int64_t addr, old_pos, row;
        bool surv;
    };
    std::vector<E> ent;
    for (int64_t i = 0; i < active; ++i) {
        ent.push_back({surv_addr[i], i, -1, true});
    }
    for (int64_t r = 0; r < R; ++r) {
        ent.push_back({r, r, r, false});
    }
    std::stable_sort(ent.begin(), ent.end(), [](const E& x, const E& y) { return x.addr < y.addr; });
    const int64_t N = static_cast<int64_t>(ent.size());
    pulsar::RecompactPlan plan;
    plan.survivor.resize(N);
    plan.old_pos.resize(N);
    plan.old_rope.resize(N);
    plan.recalled_row.resize(N);
    plan.addr.resize(N);
    plan.tok.resize(N);
    plan.score.assign(N, 0.0);
    std::vector<int64_t> merged_addr(N);
    for (int64_t i = 0; i < N; ++i) {
        plan.survivor[i] = ent[i].surv ? 1 : 0;
        plan.old_pos[i] = ent[i].old_pos;
        // Contiguous layout, recalled roped at its stored pos: the slot index and the
        // baked pos coincide for both kinds here.
        plan.old_rope[i] = ent[i].old_pos;
        plan.recalled_row[i] = ent[i].row;
        plan.addr[i] = ent[i].addr;
        plan.tok[i] = ent[i].addr;  // token id stand-in (addr), unconstrained here
        merged_addr[i] = ent[i].addr;
    }

    a.recompact_with(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        plan,
        rec_k,
        rec_v
    );

    // Headroom bound: active_len == active + the fitted pages' tokens, <= max_context.
    CHECK(a.active_len(0) == active + pages_that_fit * segment_size);
    CHECK(a.active_len(0) <= max_context);
    REQUIRE(a.active_len(0) == N);

    // No survivor copy: every physical survivor page is still in the permuted table.
    const auto pages_after = a.pages_of(0);
    std::unordered_set<int32_t> after(pages_after.begin(), pages_after.end());
    for (int32_t p : surv_pages_before) {
        CHECK(after.count(p) == 1);
    }

    // Correctness: each active pos holds rope(raw_k[addr], pos); V is the
    // untouched raw V (survivor V never moved, recalled V written once).
    auto new_slots = a.slot_mapping(0, 0, N).to(at::kLong);
    auto new_pos = at::arange(N, at::TensorOptions().dtype(at::kLong).device(DEV));
    for (int64_t l = 0; l < n_layers; ++l) {
        std::vector<at::Tensor> exp_k_rows, exp_v_rows;
        for (int64_t i = 0; i < N; ++i) {
            const int64_t ab = merged_addr[i];
            if (ab >= 48) {
                exp_k_rows.push_back(surv_raw_k[l][ab - 48]);
                exp_v_rows.push_back(surv_raw_v[l][ab - 48]);
            } else {
                exp_k_rows.push_back(rec_raw_k[l][ab]);
                exp_v_rows.push_back(rec_raw_v[l][ab]);
            }
        }
        auto got_k = a.k_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
        auto exp_k = rope_rows(at::stack(exp_k_rows), new_pos);
        CHECK(at::allclose(got_k, exp_k, 1e-3, 1e-3));
        auto got_v = a.v_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
        CHECK(at::equal(got_v, at::stack(exp_v_rows)));
    }
}

// The active buffer owns page-index/token-id identity; the whole-page invariant
// (page position pp holds addresses [page_id*ps, page_id*ps+len)) survives an
// evict->recall cycle: a survivor keeps its immutable page index and token ids,
// eviction drops a whole page index (gappy, still ascending), and recall re-inserts
// the page at its address rank so identity reconstructs the full sequence.
TEST_CASE("page-index/token identity holds across evict and recall", "[allocator][identity][cuda]") {
    require_cuda();
    at::manual_seed(0);
    const int64_t ps = 16, n_kv = 2, hd = 64, n_layers = 2;
    auto a = make_alloc(64, ps, n_kv, hd, n_layers);
    const int64_t ctx = 64;  // 4 full pages, addresses 0..63

    a.allocate(0, ctx);
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);
    a.check_page_aligned(0);
    REQUIRE(a.num_active_pages(0) == 4);
    CHECK(a.next_page_index(0) == 4);
    for (int64_t pp = 0; pp < 4; ++pp) {
        CHECK(a.page_id(0, pp) == pp);
        const auto& ids = a.page_token_ids(0, pp);
        REQUIRE(static_cast<int64_t>(ids.size()) == ps);
        for (int64_t o = 0; o < ps; ++o) {
            CHECK(ids[o] == 1000 + pp * ps + o);
        }
    }

    // Evict page index 0 deterministically: n_working=48 leaves only the first page
    // (addresses [0,16)) in the eviction band, so it is the sole candidate. Bar off
    // (min_density 0); the active_cap backstop demotes the single lowest page.
    a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/48,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.0,
        /*active_cap=*/48,
        /*block_radius=*/0,
        /*stream_len=*/ctx
    );
    REQUIRE(a.active_len(0) == 48);
    REQUIRE(a.num_active_pages(0) == 3);
    a.check_page_aligned(0);
    for (int64_t pp = 0; pp < 3; ++pp) {
        const int64_t q = pp + 1;  // survivors keep immutable page indices {1,2,3}
        CHECK(a.page_id(0, pp) == q);
        const auto& ids = a.page_token_ids(0, pp);
        REQUIRE(static_cast<int64_t>(ids.size()) == ps);
        for (int64_t o = 0; o < ps; ++o) {
            CHECK(ids[o] == 1000 + q * ps + o);
        }
    }
    CHECK(a.next_page_index(0) == 4);  // no new page created

    // Recall page index 0 back at its address rank (before the survivors).
    const int64_t R = ps;
    auto rec_pos = at::arange(R, at::TensorOptions().dtype(at::kLong).device(DEV));
    std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        rec_k[l] = rope_rows(at::randn({R, n_kv, hd}, at::device(DEV)), rec_pos);
        rec_v[l] = at::randn({R, n_kv, hd}, at::device(DEV));
    }
    const int64_t K = a.active_len(0);  // 48 survivors at addresses 16..63
    const int64_t N = K + R;  // 64, reconstructing 0..63
    pulsar::RecompactPlan plan;
    plan.survivor.assign(N, 0);
    plan.old_pos.assign(N, -1);
    plan.old_rope.assign(N, 0);
    plan.recalled_row.assign(N, -1);
    plan.addr.assign(N, 0);
    plan.tok.assign(N, 0);
    plan.score.assign(N, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        plan.addr[i] = i;  // merged temporal order is contiguous [0, 64)
        plan.tok[i] = 1000 + i;
        if (i < R) {  // addresses 0..15 recalled
            plan.survivor[i] = 0;
            plan.old_rope[i] = i;  // baked (demotion) roped pos
            plan.recalled_row[i] = i;
        } else {  // addresses 16..63 survivors
            plan.survivor[i] = 1;
            plan.old_pos[i] = i - R;  // survivor's slot index [0, 48)
            plan.old_rope[i] = i - R;  // contiguous layout: index == pos
        }
    }
    a.recompact_with(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        plan,
        rec_k,
        rec_v
    );

    REQUIRE(a.active_len(0) == N);
    REQUIRE(a.num_active_pages(0) == 4);
    a.check_page_aligned(0);
    CHECK(a.next_page_index(0) == 4);
    for (int64_t pp = 0; pp < 4; ++pp) {
        CHECK(a.page_id(0, pp) == pp);  // full sequence reconstructed
        const auto& ids = a.page_token_ids(0, pp);
        REQUIRE(static_cast<int64_t>(ids.size()) == ps);
        for (int64_t o = 0; o < ps; ++o) {
            CHECK(ids[o] == 1000 + pp * ps + o);
        }
    }
}

// Under the compacted layout a survivor's SLOT INDEX and its baked roped position are
// different numbers: every distant-region token reports short_offset while its slot
// index keeps ascending. recompact_with reads plan.old_pos as the slot index -- the
// page-alignment coordinate that finds the survivor's physical page -- and plan.old_rope
// as the rotation to undo. A plan that puts the roped position in old_pos cannot be
// page-aligned and the recompact throws.
// One evict installs the compacted layout, then the demoted page is recalled back
// against the survivors, so the merged buffer mixes both kinds. The recalled page lands
// back in the distant region, whose position IS short_offset, so its reposition delta is
// 0 and the re-rope must leave its keys exactly as demotion left them.
TEST_CASE(
    "recompact_with separates slot index from baked pos under the compacted layout",
    "[recall][recompact][rope][cuda]"
) {
    require_cuda();
    at::manual_seed(0);
    const int64_t ps = 16, n_kv = 2, hd = 64, n_layers = 2, ctx = 64;  // 4 pages
    const int64_t n_sink = 16, n_working = 16, short_offset = 57;
    auto a = make_alloc(16, ps, n_kv, hd, n_layers);
    a.allocate(0, ctx);

    // Raw keys per layer, written roped at the pre-cycle contiguous positions.
    auto pos_full = at::arange(ctx, at::TensorOptions().dtype(at::kLong).device(DEV));
    auto slots = a.slot_mapping(0, 0, ctx);
    std::vector<at::Tensor> raw_k(n_layers), raw_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        raw_k[l] = at::randn({ctx, n_kv, hd}, at::device(DEV));
        raw_v[l] = at::randn({ctx, n_kv, hd}, at::device(DEV));
        pulsar::write_kv_cuda(a.k_pool(l), a.v_pool(l), rope_rows(raw_k[l], pos_full), raw_v[l], slots);
    }
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);

    // The three-region layout a buffer of ctx_after tokens takes under these knobs.
    auto expect_rope = [&](int64_t ctx_after, int64_t index) {
        const int64_t sink = std::min(n_sink, ctx_after);
        const int64_t working_lo = std::max(sink, ctx_after - n_working);
        if (index < sink) {
            return index;
        }
        if (index < working_lo) {
            return short_offset;
        }
        return short_offset + 1 + (index - working_lo);
    };

    // Mass is all zero, so the bar demotes nothing and the active_cap backstop takes the
    // single lowest-density page, ties by position. The band [16, 48) holds whole pages
    // 1 and 2, so the victim is page 1 (addresses 16..31).
    DemotedKV d = a.evict(
        0,
        n_sink,
        n_working,
        compacted,
        short_offset,
        /*min_density=*/0.0,
        /*active_cap=*/48,
        /*block_radius=*/0,
        /*stream_len=*/ctx
    );
    REQUIRE(d.num_tokens() == ps);
    const int64_t K = a.active_len(0);
    REQUIRE(K == 48);
    // Slot index and baked position now disagree over the whole distant region.
    CHECK(a.rope_pos_at(0, 16) == short_offset);
    CHECK(a.rope_pos_at(0, 31) == short_offset);
    for (int64_t i = 0; i < K; ++i) {
        CHECK(a.rope_pos_at(0, i) == expect_rope(K, i));
    }

    // The victim's K sits roped at short_offset in its freed-but-live slots, its V
    // untouched; read both out before the recompact reclaims the page.
    std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers);
    auto demoted_pos = at::full({ps}, short_offset, at::TensorOptions().dtype(at::kLong).device(DEV));
    for (int64_t l = 0; l < n_layers; ++l) {
        rec_k[l] = a.k_pool(l).view({-1, n_kv, hd}).index_select(0, d.victim_slots).contiguous();
        rec_v[l] = a.v_pool(l).view({-1, n_kv, hd}).index_select(0, d.victim_slots).contiguous();
        CHECK(at::allclose(rec_k[l], rope_rows(raw_k[l].slice(0, ps, 2 * ps), demoted_pos), 1e-3, 1e-3));
    }

    // Merge: survivors carry their slot index AND their baked pos; the recalled page
    // carries only its stored demotion pos, short_offset. Merged addr order restores
    // [0, 64).
    std::vector<int64_t> surv_addr;
    for (int64_t i = 0; i < ctx; ++i) {
        if (i < ps || i >= 2 * ps) {
            surv_addr.push_back(i);
        }
    }
    REQUIRE(static_cast<int64_t>(surv_addr.size()) == K);
    struct E {
        int64_t addr, old_pos, old_rope, row;
        bool surv;
    };
    std::vector<E> ent;
    for (int64_t i = 0; i < K; ++i) {
        ent.push_back({surv_addr[i], i, a.rope_pos_at(0, i), -1, true});
    }
    for (int64_t r = 0; r < ps; ++r) {
        ent.push_back({ps + r, -1, short_offset, r, false});
    }
    std::stable_sort(ent.begin(), ent.end(), [](const E& x, const E& y) { return x.addr < y.addr; });
    const int64_t N = K + ps;
    pulsar::RecompactPlan plan;
    plan.survivor.resize(N);
    plan.old_pos.resize(N);
    plan.old_rope.resize(N);
    plan.recalled_row.resize(N);
    plan.addr.resize(N);
    plan.tok.resize(N);
    plan.score.assign(N, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        plan.survivor[i] = ent[i].surv ? 1 : 0;
        plan.old_pos[i] = ent[i].old_pos;
        plan.old_rope[i] = ent[i].old_rope;
        plan.recalled_row[i] = ent[i].row;
        plan.addr[i] = ent[i].addr;
        plan.tok[i] = 1000 + ent[i].addr;
        REQUIRE(ent[i].addr == i);  // merged temporal order is contiguous [0, 64)
    }
    // A plan whose old_pos carried the baked pos would throw here: the survivor page at
    // new slot 32 reports old_pos == short_offset, which is not page-aligned.
    a.recompact_with(0, n_sink, n_working, compacted, short_offset, plan, rec_k, rec_v);

    REQUIRE(a.active_len(0) == N);
    for (int64_t i = 0; i < N; ++i) {
        CHECK(a.rope_pos_at(0, i) == expect_rope(N, i));
    }
    // The recalled page lands in the distant region, so its target position is the
    // short_offset its keys already carry: a delta of 0 the reposition kernel no-ops on.
    for (int64_t i = ps; i < 2 * ps; ++i) {
        CHECK(a.rope_pos_at(0, i) == short_offset);
    }

    // Identity reconstructs the full sequence.
    REQUIRE(a.num_active_pages(0) == 4);
    a.check_page_aligned(0);
    for (int64_t pp = 0; pp < 4; ++pp) {
        CHECK(a.page_id(0, pp) == pp);
        const auto& ids = a.page_token_ids(0, pp);
        REQUIRE(static_cast<int64_t>(ids.size()) == ps);
        for (int64_t o = 0; o < ps; ++o) {
            CHECK(ids[o] == 1000 + pp * ps + o);
        }
    }

    // Every slot holds rope(raw_k[addr], layout pos), survivor and recalled alike, and
    // V is the untouched raw V.
    std::vector<int64_t> want_pos(N);
    for (int64_t i = 0; i < N; ++i) {
        want_pos[i] = expect_rope(N, i);
    }
    auto want_pos_dev = at::tensor(want_pos, at::TensorOptions().dtype(at::kLong)).to(DEV);
    auto new_slots = a.slot_mapping(0, 0, N).to(at::kLong);
    for (int64_t l = 0; l < n_layers; ++l) {
        auto got_k = a.k_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
        CHECK(at::allclose(got_k, rope_rows(raw_k[l], want_pos_dev), 1e-3, 1e-3));
        auto got_v = a.v_pool(l).view({-1, n_kv, hd}).index_select(0, new_slots);
        CHECK(at::equal(got_v, raw_v[l]));
    }
}

// The stored per-page mass is a decaying EMA, so a younger page (fewer forwards
// accumulated) reads artificially sparse. evict divides each page's mass by the
// bias-correction denominator bc = 1 - (1-decay)^age (decay is alpha, retention =
// 1 - decay; age from the page's ABSOLUTE address) to recover the per-forward mean
// density. Construct two full pages where the OLDER page carries HIGHER raw mass:
// with decay off the lower-raw younger page is evicted, but the age correction
// divides the younger page by a smaller bc and REVERSES the victim to the older
// page -- a decision only the correction explains.
TEST_CASE("evict bias-corrects page density by age", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, n_kv = 1, hd = 64, nl = 1, ctx = 16;  // 2 full pages
    auto a = make_alloc(16, pl, n_kv, hd, nl);
    a.allocate(0, ctx);
    // Contiguous identity: page 0 -> address 0 (older, age 16), page 1 -> address 8
    // (younger, age 8) at stream_len 16.
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);
    // Older page 0 raw mass 6, younger page 1 raw mass 5.
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    auto host = at::empty({ctx}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 1>();
    for (int64_t i = 0; i < ctx; ++i) {
        ha[i] = (i < pl) ? 6.0f : 5.0f;
    }
    auto vals = host.to(DEV);
    auto mass_flat = a.mass_pool(0).view({-1, n_kv});
    for (int hh = 0; hh < n_kv; ++hh) {
        mass_flat.index_put_({slots, hh}, vals);
    }

    // n_q_heads = 1, 1 relevance layer => density == max over query heads / bc == raw
    // / bc (no group divisor). active_cap = pl forces exactly ONE page to demote; the backstop
    // demotes the LOWEST-density page. min_density = 0.0 makes the absolute bar
    // demote nothing, so the decision is the density ordering alone.
    SECTION("decay off: no correction, lower-raw younger page evicted") {
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/pl,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/0.0
        );
        REQUIRE(d.num_tokens() == pl);
        CHECK(cpu_i32_vec(d.addr()) == std::vector<int64_t>({8, 9, 10, 11, 12, 13, 14, 15}));
        CHECK(a.page_id(0, 0) == 0);  // older page 0 survives
    }
    SECTION(
        "decay 0.1 (retention 0.9): age correction reverses the victim to the older "
        "page"
    ) {
        // 6 / bc(16) = 7.37 < 5 / bc(8) = 8.78 -> older page 0 is lower density, demotes.
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/pl,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/0.1
        );
        REQUIRE(d.num_tokens() == pl);
        CHECK(cpu_i32_vec(d.addr()) == std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}));
        CHECK(a.page_id(0, 0) == 1);  // younger page 1 survives (immutable index 1)
    }
}

// evict's cutoff is an ABSOLUTE [0, 1] keep-bar: a page demotes when its
// bias-corrected max-pooled density < min_density. Four pages with distinct
// densities; decay off and a single relevance layer/query head so density == the
// raw set value. active_cap = 0 isolates the bar from the backstop; min_density
// moves the boundary. A shorter or longer active set self-equilibrates around
// this fixed bar.
TEST_CASE("evict absolute [0,1] density bar", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, n_kv = 1, hd = 64, nl = 1, ctx = 32;  // 4 full pages
    auto a = make_alloc(16, pl, n_kv, hd, nl);
    a.allocate(0, ctx);
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);
    // Every evict below leaves attention_mass_decay at its 0 default: no bias
    // correction, so density == raw max and the bar math is exact.

    // Page densities: 0.02, 0.04, 0.06 and 0.10 (page 3, protected working window).
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    auto host = at::empty({ctx}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 1>();
    for (int64_t i = 0; i < ctx; ++i) {
        const int64_t pg = i / pl;
        ha[i] = pg == 0 ? 0.02f : pg == 1 ? 0.04f : pg == 2 ? 0.06f : 0.10f;
    }
    auto vals = host.to(DEV);
    auto mass_flat = a.mass_pool(0).view({-1, n_kv});
    for (int hh = 0; hh < n_kv; ++hh) {
        mass_flat.index_put_({slots, hh}, vals);
    }

    // n_working = 8 protects page 3; band [0, 24) => candidates {0, 1, 2}.
    // active_cap = 0 disables the backstop so ONLY the absolute bar decides.
    SECTION("min_density 0.05: pages 0 (0.02) and 1 (0.04) demote; page 2 (0.06) survives") {
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/8,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.05,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx
        );
        REQUIRE(d.num_tokens() == 2 * pl);
        CHECK(cpu_i32_vec(d.addr()).front() == 0);
        CHECK(a.page_id(0, 0) == 2);  // survivors {2, 3}: page 2 (0.06 >= 0.05) survives
    }
    SECTION("min_density 0.10: every candidate below 0.10 (0, 1, 2) demotes") {
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/8,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.10,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx
        );
        REQUIRE(d.num_tokens() == 3 * pl);
        CHECK(cpu_i32_vec(d.addr()).front() == 0);
        CHECK(a.page_id(0, 0) == 3);  // only the protected page 3 survives
    }
    SECTION("min_density 0.0: the density pass demotes nothing") {
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/8,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx
        );
        CHECK(d.num_tokens() == 0);
        CHECK(a.active_len(0) == ctx);
    }
}

namespace {

// Give seq 0 a contiguous ctx-token identity, and return the device slot index of every
// token. Callers evict at the default attention_mass_decay 0 (bias correction 1).
at::Tensor seed_identity(ActiveBuffer& a, int64_t ctx) {
    a.allocate(0, ctx);
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);
    return a.slot_mapping(0, 0, ctx).to(at::kLong);
}

// Seed seq 0 with a sequence whose per-query-head mass is mass[page][head] on every
// slot of the page, in the sole relevance layer, so a page's eviction density is
// exactly head_reduce over mass[page].
void seed_head_mass(ActiveBuffer& a, int64_t page_len, const std::vector<std::vector<float>>& mass) {
    const int64_t n_q = static_cast<int64_t>(mass.front().size());
    const int64_t ctx = static_cast<int64_t>(mass.size()) * page_len;
    auto slots = seed_identity(a, ctx);
    auto host = at::empty({ctx, n_q}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 2>();
    for (int64_t i = 0; i < ctx; ++i) {
        for (int64_t h = 0; h < n_q; ++h) {
            ha[i][h] = mass[i / page_len][h];
        }
    }
    a.mass_pool(0).view({-1, n_q}).index_put_({slots}, host.to(DEV));
}

// Seed seq 0 with a single-query-head sequence whose mass is mass[layer][page] on
// every slot of the page, so a page's eviction density is exactly the max over its
// column of per-layer values.
void seed_layer_mass(ActiveBuffer& a, int64_t page_len, const std::vector<std::vector<float>>& mass) {
    const int64_t n_layers = static_cast<int64_t>(mass.size());
    const int64_t ctx = static_cast<int64_t>(mass.front().size()) * page_len;
    auto slots = seed_identity(a, ctx);
    auto host = at::empty({ctx, 1}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 2>();
    for (int64_t l = 0; l < n_layers; ++l) {
        for (int64_t i = 0; i < ctx; ++i) {
            ha[i][0] = mass[l][i / page_len];
        }
        a.mass_pool(l).view({-1, 1}).index_put_({slots}, host.to(DEV));
    }
}

// Seed seq 0 in the sole relevance layer with per-slot, per-query-head mass
// mass[page][slot][head], so heads may peak on DIFFERENT slots of the same page.
void seed_slot_head_mass(ActiveBuffer& a, int64_t page_len, const std::vector<std::vector<std::vector<float>>>& mass) {
    const int64_t n_q = static_cast<int64_t>(mass.front().front().size());
    const int64_t ctx = static_cast<int64_t>(mass.size()) * page_len;
    auto slots = seed_identity(a, ctx);
    auto host = at::empty({ctx, n_q}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 2>();
    for (int64_t i = 0; i < ctx; ++i) {
        for (int64_t h = 0; h < n_q; ++h) {
            ha[i][h] = mass[i / page_len][i % page_len][h];
        }
    }
    a.mass_pool(0).view({-1, n_q}).index_put_({slots}, host.to(DEV));
}

}  // namespace

// The query-head axis of the eviction density reduces by MEAN (the page-slot axis stays
// max). Page 0 carries one hot head at 0.40 and three cold ones, page 1 four even heads
// at 0.20, so the mean ranks page 0 BELOW page 1 (0.10 < 0.20) even though its single
// head is the hottest anywhere. With the bar off and active_cap at one page the backstop
// demotes exactly the lower-ranked page.
TEST_CASE("evict query-head reduction ranks pages by their head mean", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, n_kv = 1, n_q = 4, ctx = 2 * pl;
    auto a = make_alloc(16, pl, n_kv, /*head_dim=*/64, /*n_layers=*/1, n_q);
    seed_head_mass(a, pl, {{0.40f, 0.0f, 0.0f, 0.0f}, {0.20f, 0.20f, 0.20f, 0.20f}});

    std::vector<float> dens;
    SECTION("the three cold heads dilute page 0, page 0 demotes") {
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/pl,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/0.0,
            &dens
        );
        REQUIRE(dens.size() == 2);
        CHECK(dens[0] == Approx(0.10f));
        CHECK(dens[1] == Approx(0.20f));
        REQUIRE(d.num_tokens() == pl);
        CHECK(cpu_i32_vec(d.addr()).front() == 0);
        CHECK(a.num_active_pages(0) == 1);
        CHECK(a.page_id(0, 0) == 1);
    }
    // The absolute keep-bar reads the same reduction: at 0.15 page 1 clears it on 0.20
    // and page 0 does not on 0.10.
    SECTION("at bar 0.15: page 1 survives") {
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.15,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx
        );
        CHECK(a.num_active_pages(0) == 1);
        CHECK(a.page_id(0, 0) == 1);
    }
}

// A reference length is the constant the attention kernels did NOT apply, so evict
// multiplies it back in: every candidate's density is the stored mass times the
// reference, and the bar is read in those same units. The default 0 leaves the stored
// mass alone, so it must reproduce the seeded value exactly.
TEST_CASE("evict states the density against the mass reference length", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl, reference = 4096;
    auto densities = [&](int64_t reference_length) {
        auto a = make_alloc(
            16,
            pl,
            /*n_kv_heads=*/1,
            /*head_dim=*/64,
            /*n_layers=*/1,
            /*n_q_heads=*/1
        );
        a.set_mass_reference_length(reference_length);
        seed_head_mass(a, pl, {{0.25f}, {0.75f}});
        std::vector<float> dens;
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/0.0,
            &dens
        );
        CHECK(a.num_active_pages(0) == 2);  // the bar is off: nothing demotes
        return dens;
    };
    const auto by_default = densities(0);
    REQUIRE(by_default.size() == 2);
    CHECK(by_default[0] == Approx(0.25f));
    CHECK(by_default[1] == Approx(0.75f));

    const auto scaled = densities(reference);
    REQUIRE(scaled.size() == 2);
    CHECK(scaled[0] == Approx(0.25f * reference));
    CHECK(scaled[1] == Approx(0.75f * reference));

    // The bar acts on the SCALED value, so a bar that keeps both pages by default
    // still keeps both, and one above the scaled densities takes both.
    auto demoted_at = [&](int64_t reference_length, double min_density) {
        auto a = make_alloc(
            16,
            pl,
            /*n_kv_heads=*/1,
            /*head_dim=*/64,
            /*n_layers=*/1,
            /*n_q_heads=*/1
        );
        a.set_mass_reference_length(reference_length);
        seed_head_mass(a, pl, {{0.25f}, {0.75f}});
        return a
            .evict(
                0,
                /*n_sink=*/0,
                /*n_working=*/0,
                contiguous,
                SHORT_OFFSET,
                min_density,
                /*active_cap=*/0,
                /*block_radius=*/0,
                /*stream_len=*/ctx,
                /*attention_mass_decay=*/0.0
            )
            .num_tokens();
    };
    CHECK(demoted_at(reference, /*min_density=*/0.5) == 0);
    CHECK(demoted_at(0, /*min_density=*/0.5) == pl);  // page 0 alone
    CHECK(demoted_at(reference, /*min_density=*/0.75 * reference + 1.0) == 2 * pl);
}

// One query head: the head mean is an identity, so a page reads back the mass it was
// seeded with.
TEST_CASE("evict reads the seeded density on a single query head", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl;
    auto a = make_alloc(
        16,
        pl,
        /*n_kv_heads=*/1,
        /*head_dim=*/64,
        /*n_layers=*/1,
        /*n_q_heads=*/1
    );
    seed_head_mass(a, pl, {{0.10f}, {0.30f}});
    std::vector<float> dens;
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.20,
        /*active_cap=*/0,
        /*block_radius=*/0,
        /*stream_len=*/ctx,
        /*attention_mass_decay=*/0.0,
        &dens
    );
    CHECK(d.num_tokens() == pl);
    CHECK(cpu_i32_vec(d.addr()).front() == 0);  // page 0 (0.10) is the victim
    CHECK(a.page_id(0, 0) == 1);
    REQUIRE(dens.size() == 2);
    CHECK(dens[0] == Approx(0.10f));
    CHECK(dens[1] == Approx(0.30f));
}

// The RELEVANCE-LAYER axis is always max, whatever the head reduction. One query head,
// two relevance layers: page 0 reads 0.40 in layer 0 and nothing in layer 1, page 1
// reads 0.24 in both, so BOTH modes rank page 0 above page 1 on layer 0 alone. A mean
// over the layers would read page 0 at 0.20 and flip the ranking.
TEST_CASE("evict reduces the relevance layers by max under either head reduction", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl;
    // The one hot layer carries page 0, so page 1 is the backstop's victim. The absolute
    // keep-bar reads the same reduction: page 0 clears 0.30 on its hot layer alone.
    auto run = [&](double min_density, int64_t active_cap) {
        auto a = make_alloc(
            16,
            pl,
            /*n_kv_heads=*/1,
            /*head_dim=*/64,
            /*n_layers=*/2,
            /*n_q_heads=*/1
        );
        seed_layer_mass(a, pl, {{0.40f, 0.24f}, {0.0f, 0.24f}});
        std::vector<float> dens;
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            min_density,
            active_cap,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/0.0,
            &dens
        );
        CHECK(a.num_active_pages(0) == 1);
        CHECK(a.page_id(0, 0) == 0);
        return dens;
    };
    for (const auto& dens :
         {run(/*min_density=*/0.0, /*active_cap=*/pl), run(/*min_density=*/0.30, /*active_cap=*/0)}) {
        REQUIRE(dens.size() == 2);
        CHECK(dens[0] == Approx(0.40f));
        CHECK(dens[1] == Approx(0.24f));
    }
}

// One relevance layer is the degenerate case where the layer reduction is an identity.
// Layer 0 would rank page 0 first if it counted; set_relevance_layers leaves only
// layer 1, which ranks page 1 first.
TEST_CASE("evict scores over the relevance layers alone", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl;
    auto a = make_alloc(
        16,
        pl,
        /*n_kv_heads=*/1,
        /*head_dim=*/64,
        /*n_layers=*/2,
        /*n_q_heads=*/1
    );
    seed_layer_mass(a, pl, {{0.90f, 0.0f}, {0.10f, 0.30f}});
    a.set_relevance_layers({1});
    std::vector<float> dens;
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.20,
        /*active_cap=*/0,
        /*block_radius=*/0,
        /*stream_len=*/ctx,
        /*attention_mass_decay=*/0.0,
        &dens
    );
    CHECK(d.num_tokens() == pl);
    CHECK(cpu_i32_vec(d.addr()).front() == 0);  // page 0 (0.10) is the victim
    CHECK(a.page_id(0, 0) == 1);
    REQUIRE(dens.size() == 2);
    CHECK(dens[0] == Approx(0.10f));
    CHECK(dens[1] == Approx(0.30f));
}

// The head axis is reduced BEFORE the slot axis, so a page's density is one TOKEN's
// cross-head score, not a blend of per-head favourites. Page 0: head 0 holds 0.40 on
// slot 0 and head 1 holds 0.20 on slot 5, nothing else; page 1 holds 0.25 on every slot
// of both heads. Page 0 reads mean(0.40, 0) == 0.20 on slot 0 and mean(0, 0.20) == 0.10
// on slot 5, so 0.20, BELOW page 1, and page 0 demotes; reducing the slots first would
// read mean(0.40, 0.20) == 0.30 and flip the ranking.
TEST_CASE("evict reduces the query heads before the page slots", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl;
    std::vector<std::vector<float>> hot(pl, {0.0f, 0.0f});
    hot[0][0] = 0.40f;
    hot[5][1] = 0.20f;
    const std::vector<std::vector<float>> even(pl, {0.25f, 0.25f});
    auto a = make_alloc(
        16,
        pl,
        /*n_kv_heads=*/1,
        /*head_dim=*/64,
        /*n_layers=*/1,
        /*n_q_heads=*/2
    );
    seed_slot_head_mass(a, pl, {hot, even});
    std::vector<float> dens;
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.0,
        /*active_cap=*/pl,
        /*block_radius=*/0,
        /*stream_len=*/ctx,
        /*attention_mass_decay=*/0.0,
        &dens
    );
    REQUIRE(d.num_tokens() == pl);
    CHECK(cpu_i32_vec(d.addr()).front() == 0);
    CHECK(a.page_id(0, 0) == 1);
    REQUIRE(dens.size() == 2);
    CHECK(dens[0] == Approx(0.20f));
    CHECK(dens[1] == Approx(0.25f));
}

// Contiguity is required, not merely preferred: evict narrows mass_pools to the
// relevance range, and narrow is a view only over adjacent layers.
TEST_CASE("set_relevance_layers requires a contiguous ascending run", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(
        16,
        8,
        /*n_kv_heads=*/1,
        /*head_dim=*/64,
        /*n_layers=*/4,
        /*n_q_heads=*/1
    );
    CHECK_THROWS(a.set_relevance_layers({0, 2}));
    CHECK_THROWS(a.set_relevance_layers({2, 1}));
    CHECK_THROWS(a.set_relevance_layers({1, 2, 4}));
    CHECK_NOTHROW(a.set_relevance_layers({2}));
    CHECK_NOTHROW(a.set_relevance_layers({1, 2, 3}));
}

// A single relevance layer at a NONZERO offset: only that layer's mass ranks the
// pages. Layer 0 would keep page 0 and layer 2 would keep page 1; layer 1 alone reads
// page 0 at 0.10 and page 1 at 0.30, so page 0 falls below the bar.
TEST_CASE("evict reads one relevance layer at a nonzero offset", "[eviction][cuda]") {
    require_cuda();
    const int64_t pl = 8, ctx = 2 * pl;
    auto a = make_alloc(
        16,
        pl,
        /*n_kv_heads=*/1,
        /*head_dim=*/64,
        /*n_layers=*/3,
        /*n_q_heads=*/1
    );
    seed_layer_mass(a, pl, {{0.90f, 0.0f}, {0.10f, 0.30f}, {0.0f, 0.90f}});
    a.set_relevance_layers({1});
    std::vector<float> dens;
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.20,
        /*active_cap=*/0,
        /*block_radius=*/0,
        /*stream_len=*/ctx,
        /*attention_mass_decay=*/0.0,
        &dens
    );
    REQUIRE(dens.size() == 2);
    CHECK(dens[0] == Approx(0.10f));
    CHECK(dens[1] == Approx(0.30f));
    CHECK(d.num_tokens() == pl);
    CHECK(cpu_i32_vec(d.addr()).front() == 0);
    CHECK(a.page_id(0, 0) == 1);
}

namespace {

constexpr int64_t SPL = 8;  // page size for the neighbourhood tests

// Seed seq 0 with dens.size() FULL pages: contiguous identity (page j holds
// addresses [j*SPL, (j+1)*SPL)) and every slot of page j set to dens[j] in the sole
// layer/query head, so the per-page eviction density (max over relevance layers,
// query heads and slots) equals dens[j] whenever the caller evicts at
// attention_mass_decay 0 (bias correction 1).
void seed_pages(ActiveBuffer& a, const std::vector<float>& dens) {
    const int64_t ctx = static_cast<int64_t>(dens.size()) * SPL;
    a.allocate(0, ctx);
    std::vector<int64_t> addr(ctx), toks(ctx);
    for (int64_t i = 0; i < ctx; ++i) {
        addr[i] = i;
        toks[i] = 1000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, addr, toks);
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    auto host = at::empty({ctx}, at::TensorOptions().dtype(at::kFloat));
    auto ha = host.accessor<float, 1>();
    for (int64_t i = 0; i < ctx; ++i) {
        ha[i] = dens[i / SPL];
    }
    a.mass_pool(0).view({-1, 1}).index_put_({slots, 0}, host.to(DEV));
}

// The immutable page indices seq 0 still holds, in page order.
std::vector<int64_t> survivor_pages(const ActiveBuffer& a) {
    std::vector<int64_t> v;
    for (int64_t pp = 0; pp < a.num_active_pages(0); ++pp) {
        v.push_back(a.page_id(0, pp));
    }
    return v;
}

// Which page indices seed_pages laid down are NOT in `survivors` -- the victim set,
// derived from the surviving identity rather than from evict's own bookkeeping.
std::vector<int64_t> victim_pages(const ActiveBuffer& a, int64_t n_pages) {
    const auto surv = survivor_pages(a);
    std::vector<int64_t> v;
    for (int64_t j = 0; j < n_pages; ++j) {
        if (std::find(surv.begin(), surv.end(), j) == surv.end()) {
            v.push_back(j);
        }
    }
    return v;
}

}  // namespace

// block_radius > 0 replaces each page's density with the MAX over the window
// k = 2*radius+1 in address order, so a page survives exactly when SOME page within the
// radius clears the bar. Densities 0.01 0.01 0.09 0.01 0.01, radius 1, bar 0.05.
// Neighbourhood maxima: p0 0.01, p1 0.09, p2 0.09, p3 0.09, p4 0.01. The one hot page
// carries its two neighbours over the bar and nothing else, so the survivor set is a run
// of three -- never the lone page the raw density would keep.
TEST_CASE("evict keeps a hot page's whole neighbourhood, not the page alone", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.01f, 0.01f, 0.09f, 0.01f, 0.01f});
    const int64_t ctx = 5 * SPL;

    // Whole sequence is the eviction band (no sink, no working window) and the
    // backstop is off, so the absolute bar alone decides.
    std::vector<float> cand_dens;
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.05,
        /*active_cap=*/0,
        /*block_radius=*/1,
        /*stream_len=*/ctx,
        /*attention_mass_decay=*/0.0,
        &cand_dens
    );

    REQUIRE(cand_dens.size() == 5);
    CHECK(cand_dens[2] > 0.08f);  // the hot page carries its own 0.09 ...
    CHECK(cand_dens[1] > 0.08f);  // ... and so does each neighbour inside the radius
    CHECK(cand_dens[3] > 0.08f);
    CHECK(cand_dens[0] < 0.05f);  // one page further out sees nothing
    CHECK(victim_pages(a, 5) == std::vector<int64_t>({0, 4}));
    CHECK(survivor_pages(a) == std::vector<int64_t>({1, 2, 3}));
    CHECK(a.active_len(0) == 3 * SPL);
    REQUIRE(d.num_tokens() == 2 * SPL);
    CHECK(cpu_i32_vec(d.addr()).front() == 0);
    CHECK(cpu_i32_vec(d.addr()).back() == 5 * SPL - 1);
}

// A page under the bar between two hot ones survives, and the reach stops at the radius.
// Densities 0.09 0.01 0.09 0.01 0.01, radius 1, bar 0.05. Neighbourhood maxima:
// p0 0.09, p1 0.09, p2 0.09, p3 0.09, p4 0.01 -- p3 is inside p2's radius, p4 is not.
TEST_CASE("evict neighbourhood reach stops at the radius", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.09f, 0.01f, 0.09f, 0.01f, 0.01f});
    const int64_t ctx = 5 * SPL;

    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.05,
        /*active_cap=*/0,
        /*block_radius=*/1,
        /*stream_len=*/ctx
    );

    CHECK(victim_pages(a, 5) == std::vector<int64_t>({4}));
    CHECK(survivor_pages(a) == std::vector<int64_t>({0, 1, 2, 3}));
    CHECK(a.active_len(0) == 4 * SPL);
    REQUIRE(d.num_tokens() == SPL);
    CHECK(cpu_i32_vec(d.addr()).front() == 4 * SPL);
}

// A wider radius maxes over a superset, so the neighbourhood density is monotone
// non-decreasing in the radius and the victim sets NEST: v2 inside v1 inside v0.
// Densities 0.01 0.01 0.08 0.01 0.01 0.11, bar 0.05.
//   radius 0 (raw): below the bar -> {0, 1, 3, 4}.
//   radius 1: maxima 0.01 0.08 0.08 0.08 0.11 0.11 -> {0}.
//   radius 2: maxima 0.08 0.08 0.08 0.11 0.11 0.11 -> {}.
TEST_CASE("evict victims nest as the block radius widens", "[eviction][cuda]") {
    require_cuda();
    const std::vector<float> dens = {0.01f, 0.01f, 0.08f, 0.01f, 0.01f, 0.11f};
    const int64_t ctx = 6 * SPL;
    // evict mutates the buffer, so each radius runs on its own freshly seeded copy.
    auto victims_at = [&](int64_t radius) {
        auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
        seed_pages(a, dens);
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.05,
            /*active_cap=*/0,
            /*block_radius=*/radius,
            /*stream_len=*/ctx
        );
        return victim_pages(a, 6);
    };

    const auto v0 = victims_at(0);
    const auto v1 = victims_at(1);
    const auto v2 = victims_at(2);
    CHECK(v0 == std::vector<int64_t>({0, 1, 3, 4}));
    CHECK(v1 == std::vector<int64_t>({0}));
    CHECK(v2.empty());
    // victim_pages returns ascending page indices, so std::includes decides the subset.
    CHECK(std::includes(v0.begin(), v0.end(), v1.begin(), v1.end()));
    CHECK(std::includes(v1.begin(), v1.end(), v2.begin(), v2.end()));
}

// The neighbourhood max must not break the capacity guarantee: with the bar off,
// active_cap still forces enough eviction that active_len <= active_cap, taking the
// LOWEST neighbourhood densities. Densities 0.10 0.02 0.60 0.04 0.08, radius 1.
// Neighbourhood maxima: p0 0.10, p1 0.60, p2 0.60, p3 0.60, p4 0.08. ctx = 40 and
// active_cap = 24 needs ceil(16/8) = 2 pages; the two lowest are p4 (0.08) and p0 (0.10)
// -- NOT the raw-lowest p1 (0.02) and p3 (0.04), which their neighbourhoods lift.
TEST_CASE("evict active_cap backstop still bounds active_len under the neighbourhood max", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.10f, 0.02f, 0.60f, 0.04f, 0.08f});
    const int64_t ctx = 5 * SPL;
    const int64_t cap = 3 * SPL;

    // min_density 0.0 puts every candidate above the bar, so the backstop alone acts.
    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.0,
        /*active_cap=*/cap,
        /*block_radius=*/1,
        /*stream_len=*/ctx
    );

    CHECK(victim_pages(a, 5) == std::vector<int64_t>({0, 4}));
    CHECK(survivor_pages(a) == std::vector<int64_t>({1, 2, 3}));
    CHECK(a.active_len(0) <= cap);
    CHECK(a.active_len(0) == cap);
    CHECK(d.num_tokens() == 2 * SPL);
}

// The compacted layout collapses the DISTANT region -- everything between the sink and
// the working window -- onto one RoPE position, the InfLLM/LM-Infinite layout. Six pages
// of 8, n_sink = 8 (page 0) and n_working = 8 (page 5), so pages 1..4 are evictable;
// densities 0.9 0.9 0.01 0.01 0.9 0.9 against a 0.05 bar demote pages 2 and 3, leaving
// ctx = 32. With short_offset = 8 the surviving layout is
//   sink     index 0..7   -> 0..7
//   distant  index 8..23  -> 8
//   working  index 24..31 -> 9..16
// and the next appended token continues the working run at 17. The position stays
// bounded by short_offset + n_working however long the conversation runs.
TEST_CASE("the compacted layout collapses the distant region onto one position", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.9f, 0.9f, 0.01f, 0.01f, 0.9f, 0.9f});
    const int64_t ctx = 6 * SPL;

    // short_offset == n_sink: the distant blob sits flush after the sink.
    a.evict(
        0,
        /*n_sink=*/SPL,
        /*n_working=*/SPL,
        compacted,
        /*short_offset=*/SPL,
        /*min_density=*/0.05,
        /*active_cap=*/0,
        /*block_radius=*/0,
        /*stream_len=*/ctx
    );

    REQUIRE(a.active_len(0) == 4 * SPL);
    CHECK(survivor_pages(a) == std::vector<int64_t>({0, 1, 4, 5}));
    for (int64_t i = 0; i < SPL; ++i) {
        CHECK(a.rope_pos_at(0, i) == i);  // sink
    }
    for (int64_t i = SPL; i < 3 * SPL; ++i) {  // distant
        CHECK(a.rope_pos_at(0, i) == SPL);
    }
    for (int64_t i = 3 * SPL; i < 4 * SPL; ++i) {  // working
        CHECK(a.rope_pos_at(0, i) == SPL + 1 + (i - 3 * SPL));
    }
    // At active_len the accessor gives the position the next appended token takes.
    CHECK(a.rope_pos_at(0, 4 * SPL) == 2 * SPL + 1);
}

// The contiguous layout is the identity: the slot index IS the RoPE position, before and
// after an eviction, whatever short_offset the demotion rotates victims to. This is the
// layout every other test in this file runs under.
TEST_CASE("the contiguous layout ropes the buffer by slot index", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.9f, 0.9f, 0.01f, 0.01f, 0.9f, 0.9f});
    const int64_t ctx = 6 * SPL;

    for (int64_t i = 0; i <= ctx; ++i) {
        CHECK(a.rope_pos_at(0, i) == i);
    }
    a.evict(
        0,
        /*n_sink=*/SPL,
        /*n_working=*/SPL,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.05,
        /*active_cap=*/0,
        /*block_radius=*/0,
        /*stream_len=*/ctx
    );
    REQUIRE(a.active_len(0) == 4 * SPL);
    for (int64_t i = 0; i <= 4 * SPL; ++i) {
        CHECK(a.rope_pos_at(0, i) == i);
    }
}

// A buffer that only GROWS still moves positions: the working run slides forward and
// everything it leaves behind belongs at short_offset. A caller that demotes nothing and
// recalls nothing installs no layout of its own, so it must relayout or the working run
// swallows the whole buffer. Three growth steps, no eviction anywhere.
TEST_CASE("relayout repairs the compacted layout when nothing is evicted", "[eviction][rope][cuda]") {
    require_cuda();
    at::manual_seed(0);
    const int64_t ps = 16, n_kv = 1, hd = 64, n_layers = 1;
    const int64_t n_sink = ps, n_working = ps, short_offset = 100;
    auto a = make_alloc(16, ps, n_kv, hd, n_layers);
    const int64_t total = 5 * ps;
    at::Tensor raw_k = at::randn({total, n_kv, hd}, at::device(DEV));

    // Append-time write: tokens [lo, hi) take the positions the installed layout gives
    // them, which is what the forward pass ropes a fresh chunk at.
    auto write_range = [&](int64_t lo, int64_t hi) {
        std::vector<int64_t> pos;
        for (int64_t i = lo; i < hi; ++i) {
            pos.push_back(a.rope_pos_at(0, i));
        }
        auto pos_dev = at::tensor(pos, at::TensorOptions().dtype(at::kLong)).to(DEV);
        pulsar::write_kv_cuda(
            a.k_pool(0),
            a.v_pool(0),
            rope_rows(raw_k.slice(0, lo, hi), pos_dev),
            at::zeros({hi - lo, n_kv, hd}, at::device(DEV)),
            a.slot_mapping(0, lo, hi - lo)
        );
    };

    // The distant region must read short_offset and its keys must be roped there.
    auto check_layout = [&]() {
        const int64_t ctx = a.active_len(0);
        const int64_t working_lo = pulsar::ActiveBuffer::working_start(ctx, n_sink, n_working);
        REQUIRE(working_lo > n_sink);  // there IS a distant region to check
        std::vector<int64_t> want(ctx);
        for (int64_t i = 0; i < ctx; ++i) {
            want[i] = a.rope_pos_at(0, i);
        }
        for (int64_t i = n_sink; i < working_lo; ++i) {
            CHECK(want[i] == short_offset);
        }
        auto want_dev = at::tensor(want, at::TensorOptions().dtype(at::kLong)).to(DEV);
        auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
        auto got = a.k_pool(0).view({-1, n_kv, hd}).index_select(0, slots);
        CHECK(at::allclose(got, rope_rows(raw_k.slice(0, 0, ctx), want_dev), 1e-3, 1e-3));
    };

    a.allocate(0, 3 * ps);
    write_range(0, 3 * ps);
    for (int64_t grown = 3 * ps; grown <= total; grown += ps) {
        a.relayout(0, n_sink, n_working, compacted, short_offset);
        check_layout();
        if (grown == total) {
            break;
        }
        a.append(0, ps);
        write_range(grown, grown + ps);
    }
}

// Under the contiguous layout the slot index IS the position, so a relayout finds nothing
// moved: every key stays byte-identical however far the buffer has grown.
TEST_CASE("relayout is a no-op under the contiguous layout", "[eviction][rope][cuda]") {
    require_cuda();
    at::manual_seed(1);
    const int64_t ps = 16, n_kv = 1, hd = 64, n_layers = 1;
    auto a = make_alloc(16, ps, n_kv, hd, n_layers);
    const int64_t ctx = 4 * ps;
    a.allocate(0, ctx);
    auto pos = at::arange(ctx, at::TensorOptions().dtype(at::kLong).device(DEV));
    at::Tensor raw_k = at::randn({ctx, n_kv, hd}, at::device(DEV));
    pulsar::write_kv_cuda(
        a.k_pool(0),
        a.v_pool(0),
        rope_rows(raw_k, pos),
        at::zeros({ctx, n_kv, hd}, at::device(DEV)),
        a.slot_mapping(0, 0, ctx)
    );
    auto slots = a.slot_mapping(0, 0, ctx).to(at::kLong);
    auto before = a.k_pool(0).view({-1, n_kv, hd}).index_select(0, slots).clone();

    a.relayout(0, /*n_sink=*/ps, /*n_working=*/ps, contiguous, SHORT_OFFSET);
    for (int64_t i = 0; i <= ctx; ++i) {
        CHECK(a.rope_pos_at(0, i) == i);
    }
    CHECK(at::equal(a.k_pool(0).view({-1, n_kv, hd}).index_select(0, slots), before));
}

// A neighbourhood shares one density, so the backstop's ranking ties across it. Ties
// break by AGE, oldest first. Uniform densities put all five pages at the same
// neighbourhood max; the cap needs two victims and takes the two oldest pages.
TEST_CASE("evict backstop breaks neighbourhood ties by age", "[eviction][cuda]") {
    require_cuda();
    auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, {0.10f, 0.10f, 0.10f, 0.10f, 0.10f});
    const int64_t cap = 3 * SPL;

    DemotedKV d = a.evict(
        0,
        /*n_sink=*/0,
        /*n_working=*/0,
        contiguous,
        SHORT_OFFSET,
        /*min_density=*/0.0,
        /*active_cap=*/cap,
        /*block_radius=*/1,
        /*stream_len=*/5 * SPL
    );

    CHECK(victim_pages(a, 5) == std::vector<int64_t>({0, 1}));
    CHECK(survivor_pages(a) == std::vector<int64_t>({2, 3, 4}));
    CHECK(a.active_len(0) == cap);
    CHECK(d.num_tokens() == 2 * SPL);
}

// The density-by-position probe buckets a page by NORMALIZED position,
// min(15, j * 16 / P), so profiles from cycles of different active lengths are
// comparable. Every page lands in exactly one bucket: with P < 16 most buckets stay
// empty, with P > 16 every bucket fills.
TEST_CASE("DensityByPosition::bucket_of partitions the pages", "[eviction]") {
    using pulsar::DensityByPosition;
    const int64_t nb = DensityByPosition::buckets;

    auto counts = [&](int64_t P) {
        std::vector<int64_t> c(nb, 0);
        for (int64_t j = 0; j < P; ++j) {
            const int64_t b = DensityByPosition::bucket_of(j, P);
            REQUIRE(b >= 0);
            REQUIRE(b < nb);
            ++c[b];
        }
        return c;
    };

    SECTION("P < 16 leaves buckets empty but still counts every page once") {
        const auto c = counts(5);
        CHECK(std::accumulate(c.begin(), c.end(), int64_t{0}) == 5);
        // 5 pages spread to buckets 0, 3, 6, 9, 12; the other 11 stay empty.
        CHECK(std::count(c.begin(), c.end(), int64_t{1}) == 5);
        CHECK(std::count(c.begin(), c.end(), int64_t{0}) == 11);
        for (int64_t b : {0, 3, 6, 9, 12}) {
            CHECK(c[b] == 1);
        }
    }
    SECTION("P > 16 fills every bucket and still counts every page once") {
        const auto c = counts(40);
        CHECK(std::accumulate(c.begin(), c.end(), int64_t{0}) == 40);
        CHECK(std::count(c.begin(), c.end(), int64_t{0}) == 0);
        CHECK(DensityByPosition::bucket_of(0, 40) == 0);
        CHECK(DensityByPosition::bucket_of(39, 40) == nb - 1);
    }
    // P < 16 spreads the pages out, so only P >= 16 reaches the last bucket.
    SECTION("the bucket index never decreases with position") {
        for (int64_t P : {1, 3, 16, 17, 100, 4096}) {
            int64_t prev = 0;
            for (int64_t j = 0; j < P; ++j) {
                const int64_t b = DensityByPosition::bucket_of(j, P);
                CHECK(b >= prev);
                prev = b;
            }
            CHECK(DensityByPosition::bucket_of(0, P) == 0);
            CHECK(prev == (P >= nb ? nb - 1 : (P - 1) * nb / P));
        }
    }
}

// evict's probe profiles ALL P pages in address order (sink prefix and working window
// included), not just the eviction candidates: n_working protects the recent pages
// from demotion but the profile must still cover them.
TEST_CASE("evict profiles every page, not just the candidates", "[eviction][cuda]") {
    require_cuda();
    using pulsar::DensityByPosition;
    const int64_t nb = DensityByPosition::buckets;

    auto profile_of = [&](int64_t n_pages) {
        auto a = make_alloc(64, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
        seed_pages(a, std::vector<float>(n_pages, 0.5f));
        DensityByPosition prof;
        // Bar 0 and no backstop: nothing is demoted, so the probe is all this measures.
        a.evict(
            0,
            /*n_sink=*/SPL,
            /*n_working=*/SPL,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/n_pages * SPL,
            /*attention_mass_decay=*/0.0,
            /*cand_density=*/nullptr,
            &prof
        );
        return prof;
    };

    SECTION("P < 16") {
        const auto prof = profile_of(5);
        const auto& c = prof.page_count;
        CHECK(std::accumulate(c.begin(), c.end(), int64_t{0}) == 5);
        CHECK(std::count(c.begin(), c.end(), int64_t{0}) == 11);
        for (int64_t b = 0; b < nb; ++b) {
            CHECK(prof.raw_mass_sum[b] == Approx(0.5 * c[b]));
        }
    }
    SECTION("P > 16") {
        const auto prof = profile_of(40);
        const auto& c = prof.page_count;
        CHECK(std::accumulate(c.begin(), c.end(), int64_t{0}) == 40);
        CHECK(std::count(c.begin(), c.end(), int64_t{0}) == 0);
        for (int64_t b = 0; b < nb; ++b) {
            CHECK(prof.raw_mass_sum[b] == Approx(0.5 * c[b]));
        }
    }
}

// The discriminating property the probe exists to measure: the raw and bias-corrected
// profiles differ ONLY by the divide by bc = 1 - (1-decay)^age. With decay off bc == 1
// and the two profiles are identical, so any gradient is attention's own. With decay
// on bc shrinks toward the recent end, so the corrected profile is inflated there --
// a recency gradient the raw mass does not have.
//
// P == 16 puts exactly one page per bucket, so bucket b IS page b and the ratio is
// per-page: page b has age = ctx - b*SPL, bc = 1 - (1-decay)^age.
TEST_CASE("evict raw and corrected profiles differ exactly by the bias correction", "[eviction][cuda]") {
    require_cuda();
    using pulsar::DensityByPosition;
    const int64_t n_pages = DensityByPosition::buckets;
    const int64_t ctx = n_pages * SPL;

    auto profile_at = [&](double decay) {
        auto a = make_alloc(24, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
        seed_pages(a, std::vector<float>(n_pages, 0.25f));
        DensityByPosition prof;
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/ctx,
            /*attention_mass_decay=*/decay,
            /*cand_density=*/nullptr,
            &prof
        );
        return prof;
    };

    SECTION("decay off: the correction is the identity") {
        const auto prof = profile_at(0.0);
        for (int64_t b = 0; b < n_pages; ++b) {
            REQUIRE(prof.page_count[b] == 1);
            CHECK(prof.corrected_density_sum[b] == prof.raw_mass_sum[b]);
        }
    }
    SECTION("decay on: the correction manufactures a recency gradient") {
        const double decay = 0.1;
        const auto prof = profile_at(decay);
        std::vector<double> ratio(n_pages);
        for (int64_t b = 0; b < n_pages; ++b) {
            REQUIRE(prof.page_count[b] == 1);
            CHECK(prof.raw_mass_sum[b] == Approx(0.25));  // flat raw mass by construction
            const double bc = 1.0 - std::pow(1.0 - decay, static_cast<double>(ctx - b * SPL));
            CHECK(prof.corrected_density_sum[b] == Approx(0.25 / bc).epsilon(1e-5));
            ratio[b] = prof.corrected_density_sum[b] / prof.raw_mass_sum[b];
        }
        // The youngest page is inflated well above its raw mass, the oldest barely at
        // all, and the inflation grows monotonically with position.
        CHECK(ratio.back() > 1.7);
        CHECK(ratio.front() < 1.01);
        CHECK(std::is_sorted(ratio.begin(), ratio.end()));
        CHECK(std::adjacent_find(ratio.begin(), ratio.end()) == ratio.end());
    }
}

// The probe ACCUMULATES: a second evict adds to the profile rather than replacing it.
TEST_CASE("evict adds into the position profile across calls", "[eviction][cuda]") {
    require_cuda();
    pulsar::DensityByPosition prof;
    auto a = make_alloc(24, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
    seed_pages(a, std::vector<float>(5, 0.25f));

    // Bar 0 with no backstop demotes nothing, so both calls see the same 5 pages.
    auto evict_once = [&] {
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/0,
            /*block_radius=*/0,
            /*stream_len=*/5 * SPL,
            /*attention_mass_decay=*/0.0,
            /*cand_density=*/nullptr,
            &prof
        );
    };
    evict_once();
    const auto after_one = prof;
    evict_once();

    CHECK(a.num_active_pages(0) == 5);
    for (size_t b = 0; b < prof.page_count.size(); ++b) {
        CHECK(prof.page_count[b] == 2 * after_one.page_count[b]);
        CHECK(prof.raw_mass_sum[b] == Approx(2.0 * after_one.raw_mass_sum[b]));
        CHECK(prof.corrected_density_sum[b] == Approx(2.0 * after_one.corrected_density_sum[b]));
    }
    CHECK(std::accumulate(prof.page_count.begin(), prof.page_count.end(), int64_t{0}) == 10);
}

// The block radius is configured in TOKENS and resolves to PAGES, the unit both sites
// work in. n whole pages resolve to n; a partial page is rejected.
TEST_CASE("resolve_block_radius rejects a radius that is not whole pages", "[eviction]") {
    CHECK(pulsar::resolve_block_radius(3 * 16, 16) == 3);
    CHECK(pulsar::resolve_block_radius(0, 16) == 0);
    CHECK(pulsar::resolve_block_radius(512, 256) == 2);
    // A partial page is an error, not a silent 0: 15 tokens against a 16-token page
    // would round away to no neighbourhood at all.
    CHECK_THROWS(pulsar::resolve_block_radius(15, 16));
    CHECK_THROWS(pulsar::resolve_block_radius(3 * 16 + 15, 16));
    CHECK_THROWS(pulsar::resolve_block_radius(255, 256));
    CHECK_THROWS(pulsar::resolve_block_radius(-16, 16));
}

// The same eviction under the token->page conversion: a token radius of one page
// reproduces the 1-page result exactly, and 0 reproduces the raw-density result.
// Densities 0.09 0.01 0.09 0.01 0.01, bar 0.05, as in the reach test: radius 1 keeps
// p1 and p3 on their neighbours' 0.09, radius 0 evicts both.
TEST_CASE("evict at a one-page token radius matches the 1-page radius", "[eviction][cuda]") {
    require_cuda();
    const std::vector<float> dens = {0.09f, 0.01f, 0.09f, 0.01f, 0.01f};
    const int64_t ctx = 5 * SPL;

    auto evict_at = [&](int64_t radius_tokens) {
        auto a = make_alloc(16, SPL, /*n_kv_heads=*/1, /*head_dim=*/64, /*n_layers=*/1);
        seed_pages(a, dens);
        a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/0.05,
            /*active_cap=*/0,
            pulsar::resolve_block_radius(radius_tokens, SPL),
            /*stream_len=*/ctx
        );
        return victim_pages(a, 5);
    };

    CHECK(evict_at(SPL) == std::vector<int64_t>({4}));
    // 0 is the only way to ask for no neighbourhood; a partial page is rejected upstream.
    CHECK(evict_at(0) == std::vector<int64_t>({1, 3, 4}));
}

// recompact_with seeds each recalled page so its bias-corrected max-over-query-head
// eviction density lands at plan.score[page]: s(page) = score * bc(age) (no group
// factor; mass is per query head). A bar between two pages' scores therefore demotes the
// lower-scoring one and leaves the other. With decay on, s scales with age, so an OLDER
// page gets a larger raw seed at the same read density.
TEST_CASE("recompact_with seeds recalled pages at their recall score", "[recall][recompact][eviction][cuda]") {
    require_cuda();
    at::manual_seed(0);
    const int64_t pl = 16, n_kv = 1, hd = 64,
                  n_layers = 1;  // write_kv needs pl in {16,32}
    auto a = make_alloc(32, pl, n_kv, hd, n_layers);
    // 2 survivor pages (32 tokens) active at pos 0..31, addr 32..63.
    const int64_t active = 32;
    a.allocate(0, active);
    std::vector<int64_t> surv_addr(active), surv_tok(active);
    for (int64_t i = 0; i < active; ++i) {
        surv_addr[i] = 32 + i;
        surv_tok[i] = 2000 + i;
    }
    a.set_identity(0, /*start_pos=*/0, surv_addr, surv_tok);
    // Recalled: 2 pages (32 tokens) at addr 0..31, demotion pos 0..31.
    const int64_t R = 32;
    auto rec_pos = at::arange(R, at::TensorOptions().dtype(at::kLong).device(DEV));
    std::vector<at::Tensor> rec_k(n_layers), rec_v(n_layers);
    for (int64_t l = 0; l < n_layers; ++l) {
        rec_k[l] = rope_rows(at::randn({R, n_kv, hd}, at::device(DEV)), rec_pos);
        rec_v[l] = at::randn({R, n_kv, hd}, at::device(DEV));
    }
    // Merged addr order is contiguous [0, 64): recalled 0..31 at pos 0..31, survivors
    // 32..63 at pos 32..63.
    const int64_t N = active + R;  // 64
    pulsar::RecompactPlan plan;
    plan.survivor.assign(N, 0);
    plan.old_pos.assign(N, -1);
    plan.old_rope.assign(N, 0);
    plan.recalled_row.assign(N, -1);
    plan.addr.assign(N, 0);
    plan.tok.assign(N, 0);
    plan.score.assign(N, 0.0);
    const int64_t stream_len = 64;
    // Absolute keep-bar D: the FLOOR under a recalled page's seed density.
    // n_q_heads = 1, 1 relevance layer, no group divisor, so read density = stored / bc.
    const double D = 0.10;  // evict_min_density (an absolute density bar)
    // Recalled page 0 (addr 0..15) scored ABOVE the bar, page 1 (addr 16..31) below it.
    const double score_above = 2.5 * D, score_below = 0.4 * D;
    for (int64_t i = 0; i < N; ++i) {
        plan.addr[i] = i;
        if (i < R) {  // recalled
            plan.survivor[i] = 0;
            plan.old_rope[i] = i;  // baked (demotion) roped pos
            plan.recalled_row[i] = i;
            plan.tok[i] = 1000 + i;
            plan.score[i] = i < pl ? score_above : score_below;
        } else {  // survivors
            plan.survivor[i] = 1;
            plan.old_pos[i] = i - R;  // survivor slot index [0, 32)
            plan.old_rope[i] = i - R;  // contiguous layout: index == pos
            plan.tok[i] = 2000 + (i - R);
        }
    }
    // Slot 0 is recalled page 0's first slot, slot pl recalled page 1's.
    auto page_mass = [&](int64_t slot_index) {
        auto slot = a.slot_mapping(0, slot_index, 1).to(at::kLong);
        return a.mass_pool(0).view({-1, n_kv}).index_select(0, slot).item<double>();
    };

    SECTION("decay off: each page seeds at its own score") {
        a.recompact_with(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            plan,
            rec_k,
            rec_v,
            stream_len,
            /*attention_mass_decay=*/0.0
        );
        REQUIRE(a.active_len(0) == N);
        // bc == 1, so the stored seed IS the read density.
        CHECK(std::abs(page_mass(0) - score_above) < 1e-6);
        CHECK(std::abs(page_mass(pl) - score_below) < 1e-6);
        // n_working = 32 protects the 2 survivor pages, leaving the 2 recalled pages the
        // candidates. D sits between the two scores, so the bar separates them.
        DemotedKV d = a.evict(
            0,
            /*n_sink=*/0,
            /*n_working=*/32,
            contiguous,
            SHORT_OFFSET,
            /*min_density=*/D,
            /*active_cap=*/0,
            /*block_radius=*/0,
            stream_len,
            /*attention_mass_decay=*/0.0
        );
        CHECK(d.num_tokens() == pl);
        CHECK(a.active_len(0) == N - pl);
        CHECK(a.page_id(0, 0) == 0);  // the above-bar page is the one still resident
    }
    SECTION(
        "decay on: older recalled page gets a larger age-scaled seed, same read "
        "density"
    ) {
        a.recompact_with(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            plan,
            rec_k,
            rec_v,
            stream_len,
            /*attention_mass_decay=*/0.1
        );
        // Page 0 (addr 0, age 64) is OLDER than page 1 (addr 16, age 48): larger bc,
        // larger raw seed. s = score * bc(age) (no group factor).
        const double m_older = page_mass(0);
        const double m_younger = page_mass(pl);
        CHECK(m_younger > 0.0);
        CHECK(m_older > m_younger);
        auto bc = [&](int64_t age) { return 1.0 - std::pow(0.9, static_cast<double>(age)); };
        CHECK(std::abs(m_older - score_above * bc(64)) < 1e-4);
        CHECK(std::abs(m_younger - score_below * bc(48)) < 1e-4);
        // Read density = stored / bc: each page's own score.
        CHECK(std::abs(m_older / bc(64) - score_above) < 1e-4);
        CHECK(std::abs(m_younger / bc(48) - score_below) < 1e-4);
    }
    SECTION("a zero-score page enters cold") {
        for (int64_t i = pl; i < R; ++i) {
            plan.score[i] = 0.0;
        }
        a.recompact_with(
            0,
            /*n_sink=*/0,
            /*n_working=*/0,
            contiguous,
            SHORT_OFFSET,
            plan,
            rec_k,
            rec_v,
            stream_len,
            /*attention_mass_decay=*/0.0
        );
        CHECK(std::abs(page_mass(0) - score_above) < 1e-6);
        CHECK(page_mass(pl) == 0.0);
    }
}

TEST_CASE("scheduler prefill then decode", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 256);
    s.add_request(100, 20);
    s.add_request(101, 30);
    auto d = s.step();
    CHECK(list_to_vec(d.prefill_seq_ids) == std::vector<int64_t>({100, 101}));
    CHECK(d.num_prefill_tokens() == 50);
    CHECK(d.decode_seq_ids.empty());
    CHECK(cpu_i32_vec(d.prefill_cu_seqlens_q) == std::vector<int64_t>({0, 20, 50}));
    CHECK(cpu_i32_vec(d.prefill_seqlens_k) == std::vector<int64_t>({20, 30}));
    CHECK((a.active_len(100) == 20 && a.active_len(101) == 30));
    s.update(c10::List<int64_t>({100, 101}), c10::List<int64_t>({5, 6}), c10::List<bool>({false, false}));
    auto d2 = s.step();
    CHECK(list_to_vec(d2.decode_seq_ids) == std::vector<int64_t>({100, 101}));
    CHECK(d2.prefill_seq_ids.empty());
    CHECK(d2.num_decode_tokens() == 2);
    CHECK(cpu_i32_vec(d2.decode_context_lens) == std::vector<int64_t>({21, 31}));
}

TEST_CASE("scheduler mixed prefill and decode step", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 256);
    s.add_request(1, 10);
    s.step();
    s.update(c10::List<int64_t>({1}), c10::List<int64_t>({7}), c10::List<bool>({false}));
    s.add_request(2, 40);
    auto d = s.step();
    CHECK(list_to_vec(d.decode_seq_ids) == std::vector<int64_t>({1}));
    CHECK(list_to_vec(d.prefill_seq_ids) == std::vector<int64_t>({2}));
    CHECK(cpu_i32_vec(d.decode_context_lens) == std::vector<int64_t>({11}));
    CHECK(cpu_i32_vec(d.prefill_seqlens_k) == std::vector<int64_t>({40}));
    CHECK(d.decode_slot_mapping.numel() == 1);
}

TEST_CASE("scheduler chunked prefill", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 16);
    s.add_request(9, 40);
    auto d1 = s.step();
    CHECK((d1.num_prefill_tokens() == 16 && a.active_len(9) == 16));
    s.update(c10::List<int64_t>{}, c10::List<int64_t>{}, c10::List<bool>{});
    auto d2 = s.step();
    CHECK((d2.num_prefill_tokens() == 16 && a.active_len(9) == 32));
    s.update(c10::List<int64_t>{}, c10::List<int64_t>{}, c10::List<bool>{});
    auto d3 = s.step();
    CHECK((d3.num_prefill_tokens() == 8 && a.active_len(9) == 40));
    s.update(c10::List<int64_t>({9}), c10::List<int64_t>({3}), c10::List<bool>({false}));
    auto d4 = s.step();
    CHECK(list_to_vec(d4.decode_seq_ids) == std::vector<int64_t>({9}));
}

TEST_CASE("scheduler admits within page budget", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(2, 16);
    Scheduler s(a, 8, 256);
    s.add_request(1, 30);
    s.add_request(2, 10);
    auto d = s.step();
    CHECK(list_to_vec(d.prefill_seq_ids) == std::vector<int64_t>({1}));
    CHECK(s.num_waiting() == 1);
    CHECK(a.num_free_pages() == 0);
}

TEST_CASE("scheduler frees finished sequences", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 256);
    s.add_request(1, 20);
    s.step();
    CHECK(a.num_free_pages() == 30);
    s.update(c10::List<int64_t>({1}), c10::List<int64_t>({0}), c10::List<bool>({true}));
    CHECK(a.num_free_pages() == 32);
    CHECK_FALSE(a.has_seq(1));
    CHECK(s.num_running() == 0);
}

TEST_CASE("scheduler cancel drops a sequence and frees its pages", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 256);
    s.add_request(1, 20);
    s.add_request(2, 30);
    s.step();  // admit both prompts
    CHECK(s.num_running() == 2);
    CHECK(a.has_seq(1));
    CHECK(a.has_seq(2));
    const int64_t free_before = a.num_free_pages();

    s.cancel(1);
    CHECK(s.num_running() == 1);
    CHECK(s.num_waiting() == 0);
    CHECK_FALSE(s.is_running(1));
    CHECK_FALSE(a.has_seq(1));
    CHECK(a.num_free_pages() == free_before + 2);  // seq 1 held 2 pages (20 tok)
    CHECK(a.has_seq(2));  // the other sequence is untouched

    s.cancel(1);  // no-op on an unknown sequence
    CHECK(s.num_running() == 1);
}

TEST_CASE("scheduler park keeps pages; resume re-admits from active context", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(32, 16);
    Scheduler s(a, 8, 256);
    s.add_request(1, 20);
    s.step();  // admit + prefill the 20 prompt tokens; active_len -> 20
    CHECK(a.active_len(1) == 20);
    const int64_t free_before = a.num_free_pages();

    s.park(1);
    CHECK_FALSE(s.is_running(1));
    CHECK(s.num_running() == 0);
    CHECK(a.has_seq(1));  // KV kept active
    CHECK(a.num_free_pages() == free_before);  // park frees no pages

    // resume from address 20 (no recall: address == active active_len).
    s.resume(1, 20, 25);  // 5 new tokens onto the active 20
    CHECK(s.num_waiting() == 1);

    auto d = s.step();  // prefill pos 20..24 over the active context
    CHECK(list_to_vec(d.prefill_seq_ids) == std::vector<int64_t>({1}));
    CHECK(d.prefill_slot_mapping.numel() == 5);  // 5 query tokens
    CHECK(cpu_i32_vec(d.prefill_seqlens_k) == std::vector<int64_t>({25}));

    // resume requires new tokens to prefill (prompt_len > resume_addr).
    s.park(1);
    CHECK_THROWS_AS(s.resume(1, 25, 25), std::exception);  // no tokens past address 25
    CHECK(a.has_seq(1));

    // park then free (release-equivalent) returns the pages.
    a.free(1);
    CHECK_FALSE(a.has_seq(1));
}

TEST_CASE("resume is admitted when only the append pages are free", "[scheduler][cuda]") {
    require_cuda();
    auto a = make_alloc(3, 16);  // 3 pages, 48-token capacity
    Scheduler s(a, 8, 256);
    s.add_request(1, 20);
    s.step();  // prefill 20 -> 2 pages held, context 20
    CHECK(a.active_len(1) == 20);
    CHECK(a.num_free_pages() == 1);  // only one page free

    s.park(1);
    // Continue with 15 new tokens. The whole prompt (35) would need 3 pages
    // (more than the 1 free), but the seq already holds 2 and only needs 1 more,
    // so admission must go by the append growth, not a fresh allocation.
    s.resume(1, 20, 35);  // resume from address 20 (no recall)
    auto d = s.step();
    CHECK(s.is_running(1));
    CHECK(a.active_len(1) == 35);
    CHECK(d.prefill_slot_mapping.numel() == 15);  // the 15 new tokens prefilled
}

// Fixed-bin histogram maps on Stats (host-only; no CUDA). density_bin: 24 log10
// bins over [1e-4, 1e2] (a quarter of a decade each), a range that brackets parity
// with uniform attention at 1.0; recall_bin: 20 linear bins over [0, 1].
TEST_CASE("Stats histogram binning helpers", "[stats]") {
    using pulsar::density_bin;
    using pulsar::recall_bin;

    SECTION("density_bin underflow and boundaries") {
        CHECK(density_bin(0.0) == 0);  // v <= 0 -> bin 0
        CHECK(density_bin(-1.0) == 0);  // negative -> bin 0
        CHECK(density_bin(1e-5) == 0);  // v < 1e-4 -> bin 0
        CHECK(density_bin(1e-4) == 0);  // low edge -> bin 0
        CHECK(density_bin(1e-3) == 4);  // (log10(1e-3)+4)/6*24 == 4
        CHECK(density_bin(1e-1) == 12);  // (log10(1e-1)+4)/6*24 == 12
        CHECK(density_bin(1.0) == 16);  // parity with uniform
        CHECK(density_bin(1e2) == 23);  // high edge clamps to 23
        CHECK(density_bin(1e4) == 23);  // above the range clamps to 23
    }
    SECTION("recall_bin boundaries") {
        CHECK(recall_bin(0.0) == 0);  // s <= 0 -> bin 0
        CHECK(recall_bin(-0.5) == 0);  // negative -> bin 0
        CHECK(recall_bin(0.5) == 10);  // floor(0.5*20) == 10
        CHECK(recall_bin(1.0) == 19);  // s >= 1 -> bin 19
        CHECK(recall_bin(1.5) == 19);  // s > 1 clamps to 19
    }
    SECTION("Stats accumulates counts across calls") {
        pulsar::Stats st;
        ++st.density_hist[density_bin(1e1)];
        ++st.density_hist[density_bin(1e1)];
        ++st.density_hist[density_bin(1e4)];
        ++st.recall_hist[recall_bin(0.5)];
        ++st.recall_hist[recall_bin(0.0)];
        CHECK(st.density_hist[20] == 2);
        CHECK(st.density_hist[23] == 1);
        CHECK(st.recall_hist[10] == 1);
        CHECK(st.recall_hist[0] == 1);
        // Untouched bins and the reset value stay zero.
        CHECK(st.density_hist[0] == 0);
        CHECK(pulsar::Stats{}.recall_hist[10] == 0);
    }
}

TEST_CASE("the eviction and recall bars share one formula and stay independent", "[eviction][recall]") {
    SECTION("each bar reads only its own tau") {
        pulsar::SessionConfig cfg;
        CHECK(cfg.evict_tau == 0.0);  // no sentinel: the default bars nothing
        CHECK(cfg.recall_tau == 0.0);

        // Setting one leaves the other where it was, so the recall bar can open while
        // the eviction bar stays shut.
        cfg.recall_tau = 0.62;
        CHECK(cfg.evict_tau == 0.0);
        CHECK(pulsar::evict_bar_from_tau(cfg.evict_tau) == 0.0);
        CHECK(pulsar::evict_bar_from_tau(cfg.recall_tau) > 0.0);

        cfg.evict_tau = 0.7;
        CHECK(cfg.recall_tau == 0.62);
        CHECK(pulsar::evict_bar_from_tau(cfg.evict_tau) > pulsar::evict_bar_from_tau(cfg.recall_tau));
    }

    SECTION("either direction alone is reachable") {
        // Eviction only: a bar to demote under, nothing promotes back.
        CHECK(pulsar::evict_bar_from_tau(0.7) > 0.0);
        CHECK(std::isinf(pulsar::evict_bar_from_tau(1.0)));
        // Forgetting faster than recalling: the eviction bar sits above the recall bar.
        CHECK(pulsar::evict_bar_from_tau(0.7) > pulsar::evict_bar_from_tau(0.62));
    }

    SECTION("the bar is tau's odds, and the ends are degenerate") {
        CHECK(pulsar::evict_bar_from_tau(0.5) == Approx(1.0));  // parity with uniform
        CHECK(pulsar::evict_bar_from_tau(0.0) == 0.0);
        CHECK(std::isinf(pulsar::evict_bar_from_tau(1.0)));
    }
}

namespace {

// What one evict/recall drift run left in the pool, against a host-side fp64 reference
// built from the pristine pre-RoPE keys.
//
// error holds one entry per RESIDENT token, ascending:
//   || pool key - rope(pristine, the installed layout's position for its slot) ||
//   / || rope(pristine, that position) ||
// over the token's whole [n_layers, n_kv_heads, head_dim] key.
//
// one_position_error is what the SMALLEST possible position mistake, one token, costs on
// the same references: the 10th percentile over the resident tokens, low enough to be a
// floor and robust to the few tokens whose key direction happens to make that rotation
// cheap. It is what tells the two failure modes apart, since quantization drift sits well
// under it and a bookkeeping error at or over it.
struct RopeDriftRun {
    std::vector<double> error;
    double mean_error = 0.0;
    double one_position_error = 0.0;
    int64_t max_rotations = 0;
    double mean_rotations = 0.0;
    std::string report;

    double max_error() const {
        return this->error.back();
    }
    double at_quantile(double quantile) const {
        const size_t n = this->error.size();
        return this->error[std::min(n - 1, static_cast<size_t>(quantile * static_cast<double>(n)))];
    }
};

// Drive `cycles` grow/evict/recall cycles under one layout and measure what survives.
//
// Each cycle appends one page, evicts the two oldest evictable pages, and recalls both
// back in address order, so the buffer grows by exactly one page and NO token is ever
// dropped: a token's conversation address stays equal to its slot index for the whole run
// and every token written is still resident at the end.
//
// The pool is bf16, the dtype the engine serves at, so every re-rope is a lossy
// read-rotate-write round trip through it while the fp64 reference takes none.
RopeDriftRun measure_rope_drift(pulsar::PositionLayout layout, int64_t cycles) {
    // The per-token error is taken over the token's whole key, so the layer and head count
    // set how tightly both it and the one-position floor concentrate.
    const int64_t page_size = 16, n_kv_heads = 4, head_dim = 64, n_layers = 2;
    const int64_t n_sink = page_size, n_working = 4 * page_size;
    const int64_t evicted_per_cycle = 2;
    const int64_t initial_tokens = 8 * page_size;
    const int64_t total_tokens = initial_tokens + cycles * page_size;

    at::manual_seed(11);
    // One spare page beyond the final length per page the recompact claims before the
    // evict's freed pages are reused.
    auto buffer = make_alloc(
        total_tokens / page_size + 2 * evicted_per_cycle,
        page_size,
        n_kv_heads,
        head_dim,
        n_layers,
        /*n_q_heads=*/0,
        "bfloat16"
    );

    // Pristine pre-RoPE keys, kept host-side in fp64 and never written back.
    std::vector<at::Tensor> raw_key(n_layers), raw_value(n_layers), pristine(n_layers);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        raw_key[layer] = at::randn({total_tokens, n_kv_heads, head_dim}, at::device(DEV));
        raw_value[layer] = at::randn({total_tokens, n_kv_heads, head_dim}, at::device(DEV));
        pristine[layer] = raw_key[layer].to(at::kCPU, at::kDouble);
    }

    // Per conversation address: the position its key was first roped at, the position it
    // carries now, and how many times the engine has actually rotated it. The counts come
    // from the engine's own position accessor, so they cannot drift from the layout.
    std::vector<int64_t> first_position(total_tokens, -1);
    std::vector<int64_t> baked_position(total_tokens, -1);
    std::vector<int64_t> rotations(total_tokens, 0);
    auto observe = [&](int64_t address, int64_t position) {
        if (baked_position[address] != position) {
            ++rotations[address];
        }
        baked_position[address] = position;
    };

    // The conversation address a slot index holds, read from the buffer's own identity.
    auto address_at = [&](int64_t index) {
        return buffer.page_id(0, index / page_size) * page_size + index % page_size;
    };

    // Append-time write: tokens take the positions the INSTALLED layout gives their slot
    // indices, which is what the forward pass ropes a fresh chunk at.
    auto write_tokens = [&](int64_t lo, int64_t hi) {
        std::vector<int64_t> position, address(hi - lo), token_id(hi - lo);
        for (int64_t index = lo; index < hi; ++index) {
            position.push_back(buffer.rope_pos_at(0, index));
            first_position[index] = position.back();
            baked_position[index] = position.back();
            address[index - lo] = index;
            token_id[index - lo] = 1000 + index;
        }
        auto position_device = at::tensor(position, at::TensorOptions().dtype(at::kLong)).to(DEV);
        auto slots = buffer.slot_mapping(0, lo, hi - lo);
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            pulsar::write_kv_cuda(
                buffer.k_pool(layer),
                buffer.v_pool(layer),
                rope_rows(raw_key[layer].slice(0, lo, hi), position_device).to(at::kBFloat16),
                raw_value[layer].slice(0, lo, hi).to(at::kBFloat16),
                slots
            );
        }
        buffer.set_identity(0, lo, address, token_id);
    };

    buffer.allocate(0, initial_tokens);
    write_tokens(0, initial_tokens);

    for (int64_t cycle = 0; cycle < cycles; ++cycle) {
        const int64_t grown = buffer.active_len(0);
        buffer.append(0, page_size);
        write_tokens(grown, grown + page_size);

        // Mass is zero everywhere, so every candidate ties at density 0 and the
        // active_cap backstop takes the two OLDEST evictable pages, the two just past the
        // sink. A cycle that stopped demoting exactly two would stop the measurement.
        const int64_t before_evict = buffer.active_len(0);
        DemotedKV demoted = buffer.evict(
            0,
            n_sink,
            n_working,
            layout,
            SHORT_OFFSET,
            /*min_density=*/0.0,
            /*active_cap=*/before_evict - evicted_per_cycle * page_size,
            /*block_radius=*/0,
            /*stream_len=*/before_evict
        );
        REQUIRE(demoted.num_tokens() == evicted_per_cycle * page_size);

        // The victims' K sits roped at SHORT_OFFSET in their freed-but-live slots; read
        // K and V out before the recompact reclaims the pages.
        std::vector<at::Tensor> recalled_key(n_layers), recalled_value(n_layers);
        for (int64_t layer = 0; layer < n_layers; ++layer) {
            recalled_key[layer] = buffer.k_pool(layer)
                                      .view({-1, n_kv_heads, head_dim})
                                      .index_select(0, demoted.victim_slots)
                                      .contiguous();
            recalled_value[layer] = buffer.v_pool(layer)
                                        .view({-1, n_kv_heads, head_dim})
                                        .index_select(0, demoted.victim_slots)
                                        .contiguous();
        }

        const int64_t survivors = buffer.active_len(0);
        const std::vector<int64_t> demoted_address = cpu_i32_vec(demoted.addr());
        for (int64_t address : demoted_address) {
            observe(address, SHORT_OFFSET);
        }
        for (int64_t index = 0; index < survivors; ++index) {
            observe(address_at(index), buffer.rope_pos_at(0, index));
        }

        // Merge survivors and recalled tokens back into address order. A survivor carries
        // its post-evict SLOT INDEX and the position that slot bakes; a recalled token
        // carries only SHORT_OFFSET, the position demotion left it at.
        struct MergeEntry {
            int64_t address, old_pos, old_rope, recalled_row;
            bool survivor;
        };
        std::vector<MergeEntry> merged;
        merged.reserve(before_evict);
        for (int64_t index = 0; index < survivors; ++index) {
            merged.push_back({address_at(index), index, buffer.rope_pos_at(0, index), -1, true});
        }
        for (size_t row = 0; row < demoted_address.size(); ++row) {
            merged.push_back({demoted_address[row], -1, SHORT_OFFSET, static_cast<int64_t>(row), false});
        }
        std::stable_sort(merged.begin(), merged.end(), [](const MergeEntry& x, const MergeEntry& y) {
            return x.address < y.address;
        });
        REQUIRE(static_cast<int64_t>(merged.size()) == before_evict);

        pulsar::RecompactPlan plan;
        plan.survivor.resize(before_evict);
        plan.old_pos.resize(before_evict);
        plan.old_rope.resize(before_evict);
        plan.recalled_row.resize(before_evict);
        plan.addr.resize(before_evict);
        plan.tok.resize(before_evict);
        plan.score.assign(before_evict, 0.0);
        bool address_is_slot_index = true;
        for (int64_t index = 0; index < before_evict; ++index) {
            plan.survivor[index] = merged[index].survivor ? 1 : 0;
            plan.old_pos[index] = merged[index].old_pos;
            plan.old_rope[index] = merged[index].old_rope;
            plan.recalled_row[index] = merged[index].recalled_row;
            plan.addr[index] = merged[index].address;
            plan.tok[index] = 1000 + merged[index].address;
            address_is_slot_index = address_is_slot_index && merged[index].address == index;
        }
        // Nothing is dropped, so the merge rebuilds the whole stream and a token's
        // address stays its slot index for the rest of the run.
        REQUIRE(address_is_slot_index);

        buffer.recompact_with(0, n_sink, n_working, layout, SHORT_OFFSET, plan, recalled_key, recalled_value);
        REQUIRE(buffer.active_len(0) == before_evict);
        for (int64_t index = 0; index < before_evict; ++index) {
            observe(index, buffer.rope_pos_at(0, index));
        }
    }

    const int64_t resident = buffer.active_len(0);
    REQUIRE(resident == total_tokens);
    buffer.check_page_aligned(0);

    std::vector<int64_t> final_address(resident), final_position(resident);
    for (int64_t index = 0; index < resident; ++index) {
        final_address[index] = address_at(index);
        final_position[index] = buffer.rope_pos_at(0, index);
    }

    // fp64 reference: the pristine key rotated straight to the position the INSTALLED
    // layout gives its slot index. Rotations compose, so this is the same key as one
    // rotated from where it was first roped, with none of the pool's round trips.
    // rope_rotate rounds only cos/sin to fp32, four orders under the bf16 signal.
    auto address_index = at::tensor(final_address, at::TensorOptions().dtype(at::kLong));
    auto position_column = at::tensor(final_position, at::TensorOptions().dtype(at::kLong)).unsqueeze(1);
    auto slots = buffer.slot_mapping(0, 0, resident).to(at::kLong);
    std::vector<at::Tensor> pool_rows(n_layers), pristine_rows(n_layers);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        pool_rows[layer] = buffer.k_pool(layer)
                               .view({-1, n_kv_heads, head_dim})
                               .index_select(0, slots)
                               .to(at::kCPU, at::kDouble);
        pristine_rows[layer] = pristine[layer].index_select(0, address_index);
    }
    auto stored = at::stack(pool_rows);  // [n_layers, resident, n_kv_heads, head_dim]
    auto pristine_key = at::stack(pristine_rows);
    auto reference = pulsar::rope_rotate(pristine_key, position_column, THETA);
    auto one_position_off = pulsar::rope_rotate(pristine_key, position_column + 1, THETA);

    // Everything but the token axis: one relative error per token, over its whole key.
    const std::vector<int64_t> key_axes{0, 2, 3};
    auto reference_energy = reference.pow(2).sum(at::IntArrayRef(key_axes));
    auto relative = ((stored - reference).pow(2).sum(at::IntArrayRef(key_axes)) / reference_energy).sqrt().contiguous();
    auto one_position_relative = ((one_position_off - reference).pow(2).sum(at::IntArrayRef(key_axes)) /
                                  reference_energy)
                                     .sqrt()
                                     .contiguous();
    auto relative_error = relative.accessor<double, 1>();
    auto one_position_step = one_position_relative.accessor<double, 1>();

    RopeDriftRun run;
    run.error.reserve(resident);
    std::vector<double> one_position(resident);
    int64_t worst = 0, rotation_total = 0;
    double one_position_mean = 0.0;
    for (int64_t index = 0; index < resident; ++index) {
        const int64_t rotated = rotations[final_address[index]];
        run.error.push_back(relative_error[index]);
        if (relative_error[index] > relative_error[worst]) {
            worst = index;
        }
        one_position[index] = one_position_step[index];
        one_position_mean += one_position_step[index] / static_cast<double>(resident);
        rotation_total += rotated;
        run.max_rotations = std::max(run.max_rotations, rotated);
    }
    run.mean_error = std::accumulate(run.error.begin(), run.error.end(), 0.0) / static_cast<double>(resident);
    run.mean_rotations = static_cast<double>(rotation_total) / static_cast<double>(resident);
    std::sort(one_position.begin(), one_position.end());
    run.one_position_error = one_position[resident / 10];
    const double bookkeeping_floor = 0.5 * run.one_position_error;
    const int64_t outliers = std::count_if(run.error.begin(), run.error.end(), [bookkeeping_floor](double e) {
        return e > bookkeeping_floor;
    });
    std::sort(run.error.begin(), run.error.end());

    // Which position actually explains the worst token's stored key, over every position
    // the layout hands out plus the one demotion ropes to. A bookkeeping error names a
    // different position than the layout does; drift leaves the layout's own the best fit.
    std::vector<int64_t> candidate_position(final_position);
    candidate_position.push_back(SHORT_OFFSET);
    std::sort(candidate_position.begin(), candidate_position.end());
    candidate_position.erase(
        std::unique(candidate_position.begin(), candidate_position.end()),
        candidate_position.end()
    );
    auto worst_pristine = pristine_key.select(1, worst);
    auto worst_stored = stored.select(1, worst);
    const double worst_energy = reference_energy[worst].item<double>();
    double best_error = std::numeric_limits<double>::infinity();
    int64_t best_position = -1;
    for (int64_t candidate : candidate_position) {
        auto trial = pulsar::rope_rotate(
            worst_pristine,
            at::scalar_tensor(candidate, at::TensorOptions().dtype(at::kLong)),
            THETA
        );
        const double error = std::sqrt((worst_stored - trial).pow(2).sum().item<double>() / worst_energy);
        if (error < best_error) {
            best_error = error;
            best_position = candidate;
        }
    }

    std::ostringstream out;
    out << "layout=" << (layout == compacted ? "compacted" : "contiguous") << " cycles=" << cycles
        << " resident=" << resident << " page_size=" << page_size << " n_sink=" << n_sink << " n_working=" << n_working
        << " short_offset=" << SHORT_OFFSET << "\n  re-ropes per token: mean=" << run.mean_rotations
        << " max=" << run.max_rotations << "\n  relative key error vs the fp64 reference: mean=" << run.mean_error
        << " p50=" << run.at_quantile(0.50) << " p90=" << run.at_quantile(0.90) << " p99=" << run.at_quantile(0.99)
        << " max=" << run.max_error() << "\n  a one-token position mistake costs p10=" << run.one_position_error
        << " min=" << one_position.front() << " mean=" << one_position_mean << ", tokens over half of p10 ("
        << bookkeeping_floor << "): " << outliers << "\n  worst token: slot=" << worst
        << " address=" << final_address[worst] << " first roped at " << first_position[final_address[worst]]
        << ", re-roped " << rotations[final_address[worst]] << " times, the layout puts it at " << final_position[worst]
        << ", the position that best explains its key is " << best_position << " (error there " << best_error << ")";
    run.report = out.str();
    return run;
}

}  // namespace

// Every other position test in this file exercises ONE operation, so nothing measures what
// a SEQUENCE of cycles does to a key. Two questions need the sequence:
//   - after many evict/recall cycles, does every resident key still hold the position the
//     installed layout says it holds?
//   - how much has repeated re-roping degraded the keys?
//
// The pool stores ROPED keys and reposition_kv reads scalar_t, rotates in fp32 and writes
// scalar_t back, so on a bf16 pool every re-rope is a lossy round trip. Under the
// contiguous layout the position IS the slot index, so an eviction moves nearly every
// survivor and the recall moves it back: two rotations a cycle. Under the compacted layout
// a distant-region token's position is the constant short_offset, which makes the
// rotation a delta-0 kernel no-op, and a token only rotates while it is inside the working
// window. Both layouts run the same driver and the gap between their rotation counts is
// the measurement.
//
// The two failure modes separate by an order of magnitude in per-token relative error, so
// one bound tells them apart: drift is small and spread over every token, while the
// smallest position mistake the layout could make is one token, which turns the
// highest-frequency pairs by a radian and costs a fifth of the key's norm. The bounds
// below are stated as fractions of that measured one-mistake cost rather than as absolute
// error budgets, and the report names the worst token's slot, its layout position, the
// position that best explains its key, and how many tokens cleared the floor -- a whole
// page or a whole tail over the floor is a logic error, nothing over it is drift.
TEST_CASE("keys keep their layout position across a hundred re-ropes", "[eviction][rope][cuda]") {
    require_cuda();
    const int64_t cycles = 60;
    const RopeDriftRun contiguous_run = measure_rope_drift(contiguous, cycles);
    const RopeDriftRun compacted_run = measure_rope_drift(compacted, cycles);

    INFO(contiguous_run.report);
    INFO(compacted_run.report);

    // A token must have been re-roped on the order of a hundred times, or the invariant
    // below holds vacuously.
    REQUIRE(contiguous_run.max_rotations >= 100);

    for (const RopeDriftRun* run : {&contiguous_run, &compacted_run}) {
        INFO(run->report);
        // THE INVARIANT: every resident key still holds the position the installed layout
        // says it holds. The threshold is a fraction of what a one-token position mistake
        // costs, measured in the same run, so it does not encode an error budget. Rotation
        // is lossy and its magnitude is not asserted here.
        REQUIRE(run->max_error() < 0.5 * run->one_position_error);
    }
}
