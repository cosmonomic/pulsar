#include "pulsar/ops.hpp"
#include "pulsar/rope.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

// Kernel correctness against an ATen reference. Also checks that the op is
// reachable through the dispatcher, i.e. registration took effect.

TEST_CASE("rmsnorm matches ATen reference", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const double eps = 1e-6;
    auto x = at::randn({4, 512}, at::device(at::kCUDA).dtype(at::kFloat));
    auto w = at::randn({512}, at::device(at::kCUDA).dtype(at::kFloat));

    auto got = pulsar::rmsnorm_cuda(x, w, eps);

    auto ms = x.pow(2).mean(-1, /*keepdim=*/true);
    auto ref = x * at::rsqrt(ms + eps) * w;

    REQUIRE(at::allclose(got, ref, /*rtol=*/1e-5, /*atol=*/1e-5));

    // eps is what keeps an all-zero row finite; on unit-variance input it shifts the
    // result far below any tolerance, so nothing above would notice it being dropped.
    auto zeros = at::zeros({4, 512}, at::device(at::kCUDA).dtype(at::kFloat));
    REQUIRE(at::allclose(pulsar::rmsnorm_cuda(zeros, w, eps), zeros));
}

// The mass side output is a post-softmax fraction rescaled by the keys a query
// attended, so a key holding a uniform share receives exactly 1. fp32 runs the
// scalar kernels, bf16 the tensor-core ones, so each case covers both.
namespace {

constexpr int64_t kPageSize = 16;
constexpr int64_t kHeadDim = 64;
const double kScale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor mass_decay(const std::vector<float>& v) {
    return at::tensor(v, at::dtype(at::kFloat)).to(at::kCUDA);
}

// A key pool of num_pages zeroed pages. Every score is q.k == 0, so a query's
// attention is uniform over the keys it attends and each key's share is exactly
// 1 / (keys attended). One kv head, one query head.
at::Tensor zero_pool(int64_t num_pages, at::ScalarType dt) {
    return at::zeros({num_pages, kPageSize, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
}

at::Tensor zero_mass(int64_t num_pages) {
    return at::zeros({num_pages, kPageSize, 1}, at::device(at::kCUDA).dtype(at::kFloat));
}

// Per-page max mass. ActiveBuffer::evict ranks pages by this value.
double page_mass(const at::Tensor& mass, int64_t page) {
    return mass.select(0, page).max().item<double>();
}

}  // namespace

TEST_CASE("prefill mass scales by each query's own causal key count", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t ntok = 64, num_pages = ntok / kPageSize;
    const double alpha = 0.05, retention = 1.0 - alpha;

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({ntok, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto page_tables = at::arange(num_pages, at::device(at::kCUDA).dtype(at::kInt)).view({1, num_pages});
        auto cu_seqlens_q = int32_cuda({0, static_cast<int32_t>(ntok)});
        auto seqlens_k = int32_cuda({static_cast<int32_t>(ntok)});

        // Query i attends keys [0, i] and spreads 1/(i+1) over each, so its
        // contribution is alpha * retention^(ntok-1-i) * (1/(i+1)) * (i+1): the
        // query's own causal count cancels the uniform weight and every query that
        // reaches a key hands it the same alpha * retention^(ntok-1-i). Summing that
        // geometric series over the queries i >= j puts key j at
        // 1 - retention^(ntok-j). Using the sequence's key length instead of the
        // query's causal bound would leave a 1/(i+1) factor in every term (key 0 at
        // alpha*ntok * sum_i retention^(ntok-1-i)/(i+1)).
        auto normalized = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            normalized,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            kScale,
            mass_decay({static_cast<float>(alpha)})
        );
        auto got = normalized.view({-1}).to(at::kCPU);
        for (int64_t j = 0; j < ntok; ++j) {
            const double want = 1.0 - std::pow(retention, static_cast<double>(ntok - j));
            REQUIRE(std::abs(got[j].item<double>() - want) < 1e-3 * want);
        }
    }
}

TEST_CASE("normalized decode mass is invariant to the context length", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    // Two sequences over disjoint pages, both with uniform attention, so every key
    // holds the SAME share (1 / its context) of its query's attention: 16 keys for
    // sequence 0, 128 for sequence 1.
    const int64_t short_ctx = 16, long_ctx = 128;
    const int64_t max_pages = long_ctx / kPageSize;
    const int64_t num_pages = 2 * max_pages;
    const int64_t long_page = max_pages;  // sequence 1's first page

    std::vector<int32_t> tables;
    for (int32_t p = 0; p < 2 * max_pages; ++p) {
        tables.push_back(p);
    }
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto context_lens = int32_cuda({static_cast<int32_t>(short_ctx), static_cast<int32_t>(long_ctx)});

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));

        auto normalized = zero_mass(num_pages);
        pulsar::attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            normalized,
            page_tables,
            context_lens,
            kScale,
            mass_decay({1.0f, 1.0f})
        );
        const double n_short = page_mass(normalized, 0);
        const double n_long = page_mass(normalized, long_page);
        REQUIRE(std::abs(n_short - 1.0) < 1e-4);
        REQUIRE(std::abs(n_long - n_short) < 1e-4);
    }

    // The split (flash-decode) path runs its own mass kernel; pin it too.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto split = zero_mass(num_pages);
    pulsar::attn_decode_split_cuda(
        q,
        k_pool,
        v_pool,
        split,
        page_tables,
        context_lens,
        kScale,
        /*num_splits=*/2,
        mass_decay({1.0f, 1.0f})
    );
    REQUIRE(std::abs(page_mass(split, 0) - 1.0) < 1e-4);
    REQUIRE(std::abs(page_mass(split, long_page) - 1.0) < 1e-4);
}

TEST_CASE("prefill mass takes the caller's length gain", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t ntok = 64, num_pages = ntok / kPageSize;
    // alpha 1 (retention 0) leaves the chunk's LAST query alone, and it attends the
    // whole chunk, so every key holds one uniform share 1/ntok and the length the mass
    // is stated against is the only thing left in the stored value.
    const auto alpha = mass_decay({1.0f});

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({ntok, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto page_tables = at::arange(num_pages, at::device(at::kCUDA).dtype(at::kInt)).view({1, num_pages});
        auto cu_seqlens_q = int32_cuda({0, static_cast<int32_t>(ntok)});
        auto seqlens_k = int32_cuda({static_cast<int32_t>(ntok)});

        // A gain of 0 is the same launch as naming none, so the two must agree BIT for
        // bit, not within a tolerance.
        auto implied = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(q, k_pool, v_pool, implied, page_tables, cu_seqlens_q, seqlens_k, kScale, alpha);
        auto stated = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            stated,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            kScale,
            alpha,
            std::nullopt,
            /*mass_length_gain=*/0.0
        );
        REQUIRE(at::equal(implied, stated));

        auto base = implied.view({-1}).to(at::kCPU);
        for (int64_t j = 0; j < ntok; ++j) {
            REQUIRE(std::abs(base[j].item<double>() - 1.0) < 1e-3);
        }

        // Gain 1 leaves the plain softmax weight, so a caller holding a CONSTANT
        // length can multiply it in afterwards and land back on the default.
        auto plain = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            plain,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            kScale,
            alpha,
            std::nullopt,
            /*mass_length_gain=*/1.0
        );
        auto got = plain.view({-1}).to(at::kCPU);
        const double want = 1.0 / static_cast<double>(ntok);
        for (int64_t j = 0; j < ntok; ++j) {
            REQUIRE(std::abs(got[j].item<double>() - want) < 1e-3 * want);
        }

        // Any other gain is that same weight times the gain.
        auto scaled = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            scaled,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            kScale,
            alpha,
            std::nullopt,
            /*mass_length_gain=*/4.0 * ntok
        );
        auto quad = scaled.view({-1}).to(at::kCPU);
        for (int64_t j = 0; j < ntok; ++j) {
            REQUIRE(std::abs(quad[j].item<double>() - 4.0) < 1e-3 * 4.0);
        }
    }
}

TEST_CASE("decode mass takes the caller's length gain", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    // Two sequences over disjoint pages, uniform attention over 16 and 128 keys. At
    // gain 1 the stored value is the plain share, 1/16 and 1/128: a caller multiplying
    // by one fixed length afterwards reads the longer context BELOW the shorter, which
    // is exactly what each sequence's own length cancels out.
    const int64_t short_ctx = 16, long_ctx = 128;
    const int64_t max_pages = long_ctx / kPageSize;
    const int64_t num_pages = 2 * max_pages;
    const int64_t long_page = max_pages;  // sequence 1's first page

    std::vector<int32_t> tables;
    for (int32_t p = 0; p < 2 * max_pages; ++p) {
        tables.push_back(p);
    }
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto context_lens = int32_cuda({static_cast<int32_t>(short_ctx), static_cast<int32_t>(long_ctx)});
    const auto alpha = mass_decay({1.0f, 1.0f});

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));

        auto implied = zero_mass(num_pages);
        pulsar::attn_decode_cuda(q, k_pool, v_pool, implied, page_tables, context_lens, kScale, alpha);
        auto stated = zero_mass(num_pages);
        pulsar::attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            stated,
            page_tables,
            context_lens,
            kScale,
            alpha,
            std::nullopt,
            /*mass_length_gain=*/0.0
        );
        REQUIRE(at::equal(implied, stated));

        auto plain = zero_mass(num_pages);
        pulsar::attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            plain,
            page_tables,
            context_lens,
            kScale,
            alpha,
            std::nullopt,
            /*mass_length_gain=*/1.0
        );
        REQUIRE(std::abs(page_mass(plain, 0) - 1.0 / short_ctx) < 1e-4);
        REQUIRE(std::abs(page_mass(plain, long_page) - 1.0 / long_ctx) < 1e-4);
    }

    // The split (flash-decode) path runs its own mass kernel; pin it too.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto split = zero_mass(num_pages);
    pulsar::attn_decode_split_cuda(
        q,
        k_pool,
        v_pool,
        split,
        page_tables,
        context_lens,
        kScale,
        /*num_splits=*/2,
        alpha,
        std::nullopt,
        /*mass_length_gain=*/1.0
    );
    REQUIRE(std::abs(page_mass(split, 0) - 1.0 / short_ctx) < 1e-4);
    REQUIRE(std::abs(page_mass(split, long_page) - 1.0 / long_ctx) < 1e-4);
}

// The optional LSE capture must be the denominator the attention itself used, so
// exp(logit - lse) recovers a key's attention weight. With all-zero Q and K every
// logit is 0, so the logsumexp over a row's attended keys is exactly log(ctx_len) --
// a closed form that pins both the value and the per-row key count (decode rows
// attend the whole context; prefill rows only their causal prefix).
TEST_CASE("attention exposes the softmax denominator it used", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t short_ctx = 16, long_ctx = 128;
    const int64_t max_pages = long_ctx / kPageSize;
    const int64_t num_pages = 2 * max_pages;

    std::vector<int32_t> tables;
    for (int32_t p = 0; p < 2 * max_pages; ++p) {
        tables.push_back(p);
    }
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto context_lens = int32_cuda({static_cast<int32_t>(short_ctx), static_cast<int32_t>(long_ctx)});

    auto fopts = at::device(at::kCUDA).dtype(at::kFloat);
    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto mass = zero_mass(num_pages);
        auto lse = at::zeros({2, 1}, fopts);
        pulsar::attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            mass,
            page_tables,
            context_lens,
            kScale,
            mass_decay({1.0f, 1.0f}),
            lse
        );
        auto host = lse.to(at::kCPU);
        CHECK(std::abs(host[0][0].item<double>() - std::log(short_ctx)) < 1e-4);
        CHECK(std::abs(host[1][0].item<double>() - std::log(long_ctx)) < 1e-4);
    }

    // The split (flash-decode) path reduces per-slice partials in its own combine
    // kernel and writes the capture from there, so it needs pinning separately.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto mass = zero_mass(num_pages);
    auto lse = at::zeros({2, 1}, fopts);
    pulsar::attn_decode_split_cuda(
        q,
        k_pool,
        v_pool,
        mass,
        page_tables,
        context_lens,
        kScale,
        /*num_splits=*/2,
        mass_decay({1.0f, 1.0f}),
        lse
    );
    auto host = lse.to(at::kCPU);
    CHECK(std::abs(host[0][0].item<double>() - std::log(short_ctx)) < 1e-4);
    CHECK(std::abs(host[1][0].item<double>() - std::log(long_ctx)) < 1e-4);
}

// The EMA gain alpha is per sequence. Two sequences in one batched forward accumulate
// mass at their own rate, which a batch-wide scalar cannot express. Zero Q/K makes every
// query's attention uniform, so both the gain and the within-chunk retention 1 - alpha
// have a closed form: prefill key j of a chunk of n queries receives
// alpha * sum_{i >= j} (1-alpha)^(n-1-i). alpha 1 (retention 0) keeps the chunk's last
// query alone.
TEST_CASE("prefill mass decays at each sequence's own rate", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t n = kPageSize;  // one full page of queries per sequence
    const std::vector<float> alphas = {1.0f, 0.5f};

    for (auto dt : {at::kFloat, at::kBFloat16}) {  // scalar then tensor-core kernel
        auto k_pool = zero_pool(2, dt);
        auto v_pool = zero_pool(2, dt);
        auto q = at::zeros({2 * n, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto page_tables = int32_cuda({0, 1}).view({2, 1});
        auto cu_seqlens_q = int32_cuda({0, static_cast<int32_t>(n), static_cast<int32_t>(2 * n)});
        auto seqlens_k = int32_cuda({static_cast<int32_t>(n), static_cast<int32_t>(n)});

        auto mass = zero_mass(2);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            mass,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            kScale,
            mass_decay(alphas)
        );
        auto got = mass.view({2, n}).to(at::kCPU);
        for (int64_t s = 0; s < 2; ++s) {
            const double alpha = alphas[s];
            for (int64_t j = 0; j < n; ++j) {
                double graded = 0.0;
                for (int64_t i = j; i < n; ++i) {
                    graded += std::pow(1.0 - alpha, static_cast<double>(n - 1 - i));
                }
                const double want = alpha * graded;
                REQUIRE(std::abs(got[s][j].item<double>() - want) < 1e-3 * want);
            }
        }
        // The newest key ends up 2x apart, so no single scalar produces both columns.
        const double newest_0 = got[0][n - 1].item<double>();
        const double newest_1 = got[1][n - 1].item<double>();
        CHECK(std::abs(newest_0 - 2.0 * newest_1) < 1e-3 * newest_0);
    }
}

// The decode counterpart has one query per sequence, so alpha only gains (no
// within-chunk grading) and each key of sequence s lands at exactly alpha_s.
TEST_CASE("decode mass decays at each sequence's own rate", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t ctx = kPageSize;
    const std::vector<float> alphas = {1.0f, 0.25f};
    auto page_tables = int32_cuda({0, 1}).view({2, 1});
    auto context_lens = int32_cuda({static_cast<int32_t>(ctx), static_cast<int32_t>(ctx)});

    for (auto dt : {at::kFloat, at::kBFloat16}) {  // scalar then tensor-core kernel
        auto k_pool = zero_pool(2, dt);
        auto v_pool = zero_pool(2, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto mass = zero_mass(2);
        pulsar::attn_decode_cuda(q, k_pool, v_pool, mass, page_tables, context_lens, kScale, mass_decay(alphas));
        auto got = mass.view({2, ctx}).to(at::kCPU);
        for (int64_t s = 0; s < 2; ++s) {
            for (int64_t j = 0; j < ctx; ++j) {
                REQUIRE(std::abs(got[s][j].item<double>() - alphas[s]) < 1e-4);
            }
        }
    }

    // The split (flash-decode) path runs its own mass kernel; pin it too.
    auto k_pool = zero_pool(2, at::kBFloat16);
    auto v_pool = zero_pool(2, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto split = zero_mass(2);
    pulsar::attn_decode_split_cuda(
        q,
        k_pool,
        v_pool,
        split,
        page_tables,
        context_lens,
        kScale,
        /*num_splits=*/2,
        mass_decay(alphas)
    );
    auto got = split.view({2, ctx}).to(at::kCPU);
    for (int64_t s = 0; s < 2; ++s) {
        REQUIRE(std::abs(got[s][0].item<double>() - alphas[s]) < 1e-4);
    }
}

// The recall scorer rotates its query rows and candidate keys with the ATen
// pulsar::rope_rotate, but the keys it scores were stored by the rope kernel, so the two
// rotations must agree at every position the buffer reaches. An angle whose rounding
// scales with the position (an fp32 frequency table) drifts here, worst at the longest
// contexts.
TEST_CASE("ATen rope_rotate matches the rope kernel", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const double theta = 1e6;
    const int64_t n_heads = 3, head_dim = 128;
    // Past the trained context as well as inside it: the drift is proportional to the
    // position, so a bound that holds at 131072 holds everywhere below.
    const std::vector<int64_t> positions{0, 1, 1024, 32768, 131072};
    const int64_t seq = static_cast<int64_t>(positions.size());
    auto pos = at::tensor(positions, at::device(at::kCUDA).dtype(at::kLong));
    auto x = at::randn({n_heads, seq, head_dim}, at::device(at::kCUDA).dtype(at::kFloat));

    // Both round cos/sin to fp32 and rotate in fp32, so they agree to a few fp32 ulps
    // of a unit-scale input, not to the bit: the measured worst gap is 2.4e-7 for the
    // single rotation and 4.8e-7 for the delta, at every position alike. 1e-5 leaves
    // 20x headroom over that and still sits 130x under the 1.3e-3 an fp32 frequency
    // table put on the 32768 row.
    const double tol = 1e-5;
    auto worst = [](const at::Tensor& a, const at::Tensor& b) { return (a - b).abs().max().item<double>(); };

    CHECK(worst(pulsar::rope_rotate(x, pos.view({1, -1}), theta), pulsar::rope_cuda(x, pos, theta)) < tol);

    // The scorer's second rotation is a DELTA over already-roped keys, one scalar angle
    // for the whole tensor. Rotations compose, so stored + delta must land on scored.
    const int64_t stored = 128, scored = 131072;
    auto stored_pos = at::full({seq}, stored, at::device(at::kCUDA).dtype(at::kLong));
    auto delta = at::scalar_tensor(static_cast<double>(scored - stored), at::device(at::kCUDA).dtype(at::kDouble));
    CHECK(
        worst(
            pulsar::rope_rotate(pulsar::rope_cuda(x, stored_pos, theta), delta, theta),
            pulsar::rope_cuda(x, at::full_like(stored_pos, scored), theta)
        ) < tol
    );
}
