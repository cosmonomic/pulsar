#include "pulsar/ops.hpp"
#include "pulsar/rope.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
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
constexpr double kTheta = 1e6;
const double kScale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor mass_decay(const std::vector<float>& v) {
    return at::tensor(v, at::dtype(at::kFloat)).to(at::kCUDA);
}

// {0, 0, -1} per sequence: every key at its own view index.
at::Tensor identity_layout(int64_t num_seqs) {
    return int32_cuda({0, 0, -1}).repeat({num_seqs, 1});
}

// A key pool of num_pages zeroed pages. Every score is q.k == 0 at any rotation, so a
// query's attention is uniform over the keys it attends and each key's share is
// exactly 1 / (keys attended). One kv head, one query head.
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

// Every sequence's view is page_tables row s, its mass rows one view page per table
// column, so view page b of sequence s is mass row s * max_pages + b.
at::Tensor row_view_pages(const at::Tensor& page_tables) {
    const int64_t num_seqs = page_tables.size(0), max_pages = page_tables.size(1);
    return at::arange(0, (num_seqs + 1) * max_pages, max_pages, at::device(at::kCUDA).dtype(at::kInt));
}

at::Tensor prefill(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& mass,
    const at::Tensor& decay,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse = std::nullopt
) {
    return pulsar::attn_prefill_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        identity_layout(seqlens_k.size(0)),
        kTheta,
        kScale,
        mass,
        row_view_pages(page_tables),
        decay,
        mass_length_gain,
        lse
    );
}

at::Tensor decode(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& mass,
    const at::Tensor& decay,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse = std::nullopt,
    int64_t num_splits = 0
) {
    return pulsar::attn_decode_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        identity_layout(context_lens.size(0)),
        kTheta,
        kScale,
        mass,
        row_view_pages(page_tables),
        decay,
        mass_length_gain,
        lse,
        num_splits
    );
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
        prefill(
            q,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            normalized,
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
    const int64_t long_page = max_pages;  // sequence 1's first view page

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
        decode(q, k_pool, v_pool, page_tables, context_lens, normalized, mass_decay({1.0f, 1.0f}));
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
    decode(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        split,
        mass_decay({1.0f, 1.0f}),
        0.0,
        std::nullopt,
        /*num_splits=*/2
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

        // A gain of 0 is the default, so the two must agree BIT for bit, not within a
        // tolerance.
        auto implied = zero_mass(num_pages);
        pulsar::attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            identity_layout(1),
            kTheta,
            kScale,
            implied,
            row_view_pages(page_tables),
            alpha
        );
        auto stated = zero_mass(num_pages);
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, stated, alpha, /*mass_length_gain=*/0.0);
        REQUIRE(at::equal(implied, stated));

        auto base = implied.view({-1}).to(at::kCPU);
        for (int64_t j = 0; j < ntok; ++j) {
            REQUIRE(std::abs(base[j].item<double>() - 1.0) < 1e-3);
        }

        // Gain 1 leaves the plain softmax weight, so a caller holding a CONSTANT
        // length can multiply it in afterwards and land back on the default.
        auto plain = zero_mass(num_pages);
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, plain, alpha, /*mass_length_gain=*/1.0);
        auto got = plain.view({-1}).to(at::kCPU);
        const double want = 1.0 / static_cast<double>(ntok);
        for (int64_t j = 0; j < ntok; ++j) {
            REQUIRE(std::abs(got[j].item<double>() - want) < 1e-3 * want);
        }

        // Any other gain is that same weight times the gain.
        auto scaled = zero_mass(num_pages);
        prefill(
            q,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            scaled,
            alpha,
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
    const int64_t long_page = max_pages;  // sequence 1's first view page

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
        pulsar::attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            page_tables,
            context_lens,
            identity_layout(2),
            kTheta,
            kScale,
            implied,
            row_view_pages(page_tables),
            alpha
        );
        auto stated = zero_mass(num_pages);
        decode(q, k_pool, v_pool, page_tables, context_lens, stated, alpha, /*mass_length_gain=*/0.0);
        REQUIRE(at::equal(implied, stated));

        auto plain = zero_mass(num_pages);
        decode(q, k_pool, v_pool, page_tables, context_lens, plain, alpha, /*mass_length_gain=*/1.0);
        REQUIRE(std::abs(page_mass(plain, 0) - 1.0 / short_ctx) < 1e-4);
        REQUIRE(std::abs(page_mass(plain, long_page) - 1.0 / long_ctx) < 1e-4);
    }

    // The split (flash-decode) path runs its own mass kernel; pin it too.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto split = zero_mass(num_pages);
    decode(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        split,
        alpha,
        /*mass_length_gain=*/1.0,
        std::nullopt,
        /*num_splits=*/2
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
        decode(q, k_pool, v_pool, page_tables, context_lens, mass, mass_decay({1.0f, 1.0f}), 0.0, lse);
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
    decode(q, k_pool, v_pool, page_tables, context_lens, mass, mass_decay({1.0f, 1.0f}), 0.0, lse, /*num_splits=*/2);
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
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, mass, mass_decay(alphas));
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
        decode(q, k_pool, v_pool, page_tables, context_lens, mass, mass_decay(alphas));
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
    decode(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        split,
        mass_decay(alphas),
        0.0,
        std::nullopt,
        /*num_splits=*/2
    );
    auto got = split.view({2, ctx}).to(at::kCPU);
    for (int64_t s = 0; s < 2; ++s) {
        REQUIRE(std::abs(got[s][0].item<double>() - alphas[s]) < 1e-4);
    }
}

namespace {

// One sequence of the shared-lane case: its view over the pool and its RoPE layout.
struct View {
    std::vector<int32_t> lanes;
    int64_t ctx;
    int64_t seq_q;  // prefill queries, the last seq_q view indices
    std::array<int32_t, 3> layout;  // {n_sink, working_lo, short_offset}

    int64_t position(int64_t j) const {
        if (j < this->layout[0]) {
            return j;
        }
        if (j < this->layout[1]) {
            return this->layout[2];
        }
        return this->layout[2] + 1 + j - this->layout[1];
    }
};

// ATen reference for one view: its keys rotated to their layout positions by
// rope_rotate and rounded to the pool dtype as the kernels store them, then causal
// softmax attention for the query rows at view indices [ctx - rows, ctx). Fills o
// rows and adds alpha-graded, length-gained mass into the view's own rows.
void reference_view(
    const View& view,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    int64_t rows,
    int64_t q_offset,
    double alpha,
    at::Tensor& o,
    at::Tensor& mass_rows
) {
    const int64_t n_kv = k_pool.size(2), head_dim = k_pool.size(3);
    const int64_t group = q.size(1) / n_kv;
    auto lanes = at::tensor(view.lanes, at::dtype(at::kLong)).to(at::kCUDA);
    auto k = k_pool.index_select(0, lanes).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    auto v = v_pool.index_select(0, lanes).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    std::vector<int64_t> positions(view.ctx);
    for (int64_t j = 0; j < view.ctx; ++j) {
        positions[j] = view.position(j);
    }
    auto pos = at::tensor(positions, at::dtype(at::kLong)).to(at::kCUDA).view({-1, 1});
    k = pulsar::rope_rotate(k, pos, kTheta).to(k_pool.scalar_type()).to(at::kFloat);

    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = view.ctx - rows + r;
        const double graded = alpha * std::pow(1.0 - alpha, static_cast<double>(rows - 1 - r)) * (p + 1);
        for (int64_t h = 0; h < q.size(1); ++h) {
            const int64_t g = h / group;
            auto keys = k.slice(0, 0, p + 1).select(1, g);
            auto values = v.slice(0, 0, p + 1).select(1, g);
            auto w = at::softmax(at::mv(keys, q[q_offset + r][h].to(at::kFloat)) * kScale, 0);
            o[q_offset + r][h] = at::mv(values.t(), w);
            mass_rows.slice(0, 0, p + 1).select(1, h).add_(w * graded);
        }
    }
}

}  // namespace

// Two views share lane 2: a full page of the second view, the partial tail of the
// first. The first view is laid out contiguously, the second compacted, so the shared
// lane's keys sit at different positions in each and must be rotated per view, and its
// mass must land in each view's own rows. The reference rotates the unrotated pool with
// the ATen rope_rotate.
TEST_CASE("views sharing a lane rotate and accumulate apart", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t n_q = 4, n_kv = 2, num_lanes = 3;
    const std::vector<View> views = {
        View{{0, 2}, kPageSize + 5, 5, {0, 0, -1}},
        View{{1, 2, 0}, 2 * kPageSize + 7, 20, {4, 2 * kPageSize + 7 - 18, 6}},
    };
    const double alpha = 0.5;
    const int64_t max_pages = 3, total_view_pages = 2 * max_pages;

    std::vector<int32_t> tables(views.size() * max_pages, 0);
    std::vector<int32_t> layouts, seqlens, cu_q{0}, cu_view;
    for (size_t s = 0; s < views.size(); ++s) {
        std::copy(views[s].lanes.begin(), views[s].lanes.end(), tables.begin() + s * max_pages);
        layouts.insert(layouts.end(), views[s].layout.begin(), views[s].layout.end());
        seqlens.push_back(static_cast<int32_t>(views[s].ctx));
        cu_q.push_back(cu_q.back() + static_cast<int32_t>(views[s].seq_q));
        cu_view.push_back(static_cast<int32_t>(s * max_pages));
    }
    cu_view.push_back(static_cast<int32_t>(total_view_pages));
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto rope_layout = int32_cuda(layouts).view({2, 3});
    auto seqlens_k = int32_cuda(seqlens);
    auto cu_seqlens_q = int32_cuda(cu_q);
    auto cu_view_pages = int32_cuda(cu_view);
    auto decay = mass_decay({static_cast<float>(alpha), static_cast<float>(alpha)});
    auto new_mass = [&] {
        return at::zeros({total_view_pages, kPageSize, n_q}, at::device(at::kCUDA).dtype(at::kFloat));
    };

    for (auto dt : {at::kFloat, at::kBFloat16}) {  // scalar then tensor-core kernels
        at::manual_seed(7);
        auto opts = at::device(at::kCUDA).dtype(dt);
        auto k_pool = at::randn({num_lanes, kPageSize, n_kv, kHeadDim}, opts);
        auto v_pool = at::randn({num_lanes, kPageSize, n_kv, kHeadDim}, opts);
        const double o_tol = dt == at::kFloat ? 1e-4 : 2e-2;

        auto q_pre = at::randn({cu_q.back(), n_q, kHeadDim}, opts);
        auto o_pre = at::zeros({cu_q.back(), n_q, kHeadDim}, at::device(at::kCUDA).dtype(at::kFloat));
        auto mass_pre = new_mass();
        auto q_dec = at::randn({2, n_q, kHeadDim}, opts);
        auto o_dec = at::zeros({2, n_q, kHeadDim}, at::device(at::kCUDA).dtype(at::kFloat));
        auto mass_dec = new_mass();
        for (size_t s = 0; s < views.size(); ++s) {
            auto pre_rows = mass_pre.slice(0, s * max_pages, (s + 1) * max_pages).view({-1, n_q});
            reference_view(views[s], q_pre, k_pool, v_pool, views[s].seq_q, cu_q[s], alpha, o_pre, pre_rows);
            auto dec_rows = mass_dec.slice(0, s * max_pages, (s + 1) * max_pages).view({-1, n_q});
            reference_view(views[s], q_dec, k_pool, v_pool, 1, s, alpha, o_dec, dec_rows);
        }

        auto mass = new_mass();
        auto o = pulsar::attn_prefill_cuda(
            q_pre,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            rope_layout,
            kTheta,
            kScale,
            mass,
            cu_view_pages,
            decay
        );
        CHECK(at::allclose(o.to(at::kFloat), o_pre, o_tol, o_tol));
        CHECK(at::allclose(mass, mass_pre, 1e-3, 1e-3));

        for (int64_t num_splits : {0, 1, 2}) {
            auto dec_mass = new_mass();
            auto dec_o = pulsar::attn_decode_cuda(
                q_dec,
                k_pool,
                v_pool,
                page_tables,
                seqlens_k,
                rope_layout,
                kTheta,
                kScale,
                dec_mass,
                cu_view_pages,
                decay,
                0.0,
                std::nullopt,
                num_splits
            );
            CHECK(at::allclose(dec_o.to(at::kFloat), o_dec, o_tol, o_tol));
            CHECK(at::allclose(dec_mass, mass_dec, 1e-3, 1e-3));
        }
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
