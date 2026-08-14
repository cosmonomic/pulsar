#include "pulsar/runtime/sampler.hpp"

#include <ATen/ATen.h>

// torch's logging header defines a CHECK macro; drop it so Catch2's CHECK wins.
#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <set>
#include <vector>

// Pure CPU. Logits are built as log(probs) so that at temperature 1 softmax
// recovers the intended distribution, making the surviving candidate set exactly
// predictable.

namespace {

using pulsar::GenerationParams;
using pulsar::make_generator;
using pulsar::StochasticSampler;

at::Tensor logits_from_probs(std::vector<float> p) {
    return at::log(at::tensor(p, at::TensorOptions().dtype(at::kFloat)));
}

at::Generator cpu_gen(int64_t seed) {
    return make_generator(at::Device(at::kCPU), seed);
}

std::vector<int64_t>
draw(StochasticSampler& s, const at::Tensor& logits, const GenerationParams& p, at::Generator& gen, int n) {
    std::vector<int64_t> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        out.push_back(s.sample(logits, p, gen));
    }
    return out;
}

// The candidate set reached over `draws` samples. Every kept token's renormalized
// mass here is above 0.07, so 500 draws miss one with probability below e^-35: the
// set is the nucleus, not a sampling artefact.
std::set<int64_t>
reached(StochasticSampler& s, const at::Tensor& logits, const GenerationParams& p, int64_t seed, int64_t draws = 500) {
    at::Generator gen = cpu_gen(seed);
    std::set<int64_t> seen;
    for (int64_t t : draw(s, logits, p, gen, draws)) {
        seen.insert(t);
    }
    return seen;
}

}  // namespace

TEST_CASE("temperature <= 0 is greedy argmax, ignoring top-k/top-p") {
    StochasticSampler s;
    at::Tensor logits = at::tensor({0.1f, 2.0f, 0.5f, 1.0f});
    // Truncation knobs set but must be ignored on the greedy path.
    GenerationParams p{/*temperature=*/0.0,
                       /*top_p=*/0.5,
                       /*top_k=*/2,
                       /*seed=*/-1};
    at::Generator gen = cpu_gen(0);
    REQUIRE(s.sample(logits, p, gen) == 1);

    p.temperature = -1.0;  // any non-positive temperature is greedy
    REQUIRE(s.sample(logits, p, gen) == 1);
}

TEST_CASE("sampling is seed-deterministic") {
    StochasticSampler s;
    at::Tensor logits = logits_from_probs(std::vector<float>(8, 0.125f));
    GenerationParams p{/*temperature=*/1.0,
                       /*top_p=*/1.0,
                       /*top_k=*/0,
                       /*seed=*/42};

    at::Generator a = cpu_gen(42);
    at::Generator b = cpu_gen(42);
    at::Generator c = cpu_gen(43);
    std::vector<int64_t> da = draw(s, logits, p, a, 32);
    std::vector<int64_t> db = draw(s, logits, p, b, 32);
    std::vector<int64_t> dc = draw(s, logits, p, c, 32);

    REQUIRE(da == db);  // same seed -> identical draw sequence
    REQUIRE(da != dc);  // a different seed diverges (negligible collision)
}

TEST_CASE("top-p nucleus keeps the exclusive-prefix <= top_p set") {
    StochasticSampler s;
    // Sorted mass 0.5,0.3,0.1,0.07,0.03; exclusive prefixes 0,0.5,0.8,0.9,0.97.
    // top_p 0.92 keeps indices 0..3 -- index 3 is the CROSSING token, whose own mass
    // takes the running total past the bar, and it is retained because the bar is on
    // the mass BEFORE it. Index 4 sits at 0.97 and is dropped.
    at::Tensor logits = logits_from_probs({0.5f, 0.3f, 0.1f, 0.07f, 0.03f});
    GenerationParams p{/*temperature=*/1.0,
                       /*top_p=*/0.92,
                       /*top_k=*/0,
                       /*seed=*/7};
    CHECK(reached(s, logits, p, 7) == std::set<int64_t>{0, 1, 2, 3});
}

TEST_CASE("top-k caps the candidate count") {
    StochasticSampler s;
    at::Tensor logits = logits_from_probs({0.4f, 0.3f, 0.2f, 0.1f});
    GenerationParams p{/*temperature=*/1.0,
                       /*top_p=*/1.0,
                       /*top_k=*/2,
                       /*seed=*/1};
    // Both of the two highest-probability tokens are reachable and neither of the
    // other two is: an argmax-only sampler fails this as surely as an uncapped one.
    CHECK(reached(s, logits, p, 1) == std::set<int64_t>{0, 1});
}

TEST_CASE("top-k and top-p compose; the tighter bound wins") {
    StochasticSampler s;
    // Both filters keep a prefix of the descending order, so their intersection is
    // whichever prefix is shorter. Exercise it in both directions, or a fixture where
    // one filter is inert would pass with that filter deleted outright.
    // Mass 0.4,0.3,0.2,0.1; exclusive prefixes 0,0.4,0.7,0.9.
    at::Tensor logits = logits_from_probs({0.4f, 0.3f, 0.2f, 0.1f});
    // top_p 0.95 keeps all four; top_k 2 is the binding one.
    GenerationParams by_k{/*temperature=*/1.0,
                          /*top_p=*/0.95,
                          /*top_k=*/2,
                          /*seed=*/2};
    CHECK(reached(s, logits, by_k, 2) == std::set<int64_t>{0, 1});
    // top_k 3 keeps {0,1,2}; top_p 0.5 cuts at index 2's prefix 0.7 and binds instead.
    GenerationParams by_p{/*temperature=*/1.0,
                          /*top_p=*/0.5,
                          /*top_k=*/3,
                          /*seed=*/3};
    CHECK(reached(s, logits, by_p, 3) == std::set<int64_t>{0, 1});
}
