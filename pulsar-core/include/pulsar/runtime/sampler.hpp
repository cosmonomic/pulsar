#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>

#include <c10/core/Device.h>

#include <cstdint>

// The sampling interface the Engine drives: one row of logits + the per-request
// generation knobs + a per-seq RNG -> a token id.

namespace pulsar {

// Per-request sampling knobs: temperature/top_p/top_k control truncation; seed drives
// the per-seq RNG (read by the Engine, not the Sampler). temperature <= 0 is exact
// greedy (argmax). top_p in (0,1) restricts to the nucleus; top_k > 0 restricts to the
// k highest-probability tokens; when both are set the surviving set is their
// intersection. seed == -1 means unseeded. eos_id is not a sampling knob; it lives on
// Engine and Engine::finished reads it from there.
struct GenerationParams {
    double temperature = 0.0;
    double top_p = 1.0;
    int64_t top_k = 0;
    int64_t seed = -1;
};

// logits row [vocab] + params + a per-seq RNG -> token id.
struct Sampler {
    virtual int64_t sample(const at::Tensor& logits_row, const GenerationParams& params, at::Generator& gen) = 0;
    virtual ~Sampler() = default;
};

// temperature/top-k/top-p sampling. temperature <= 0 -> argmax (greedy; gen
// unused). Otherwise softmax(logits.float() / temperature), optional top-k then
// top-p (nucleus) truncation, then multinomial over the surviving mass using
// gen. Seed-deterministic through gen.
struct StochasticSampler : Sampler {
    int64_t sample(const at::Tensor& logits_row, const GenerationParams& params, at::Generator& gen) override;
};

// Build a per-seq RNG on `device`. seed >= 0 is reproducible; seed < 0 draws a
// nondeterministic seed.
at::Generator make_generator(at::Device device, int64_t seed);

}  // namespace pulsar
