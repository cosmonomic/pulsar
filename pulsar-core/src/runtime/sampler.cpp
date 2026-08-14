#include "pulsar/runtime/sampler.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <tuple>

namespace pulsar {

int64_t StochasticSampler::sample(const at::Tensor& logits_row, const GenerationParams& params, at::Generator& gen) {
    if (params.temperature <= 0.0) {
        return logits_row.argmax().item<int64_t>();
    }

    at::Tensor probs = at::softmax(logits_row.to(at::kFloat) / params.temperature, -1);

    const bool use_top_k = params.top_k > 0;
    const bool use_top_p = params.top_p > 0.0 && params.top_p < 1.0;

    if (!use_top_k && !use_top_p) {
        return at::multinomial(probs, 1, /*replacement=*/false, gen).item<int64_t>();
    }

    auto sorted = at::sort(probs, /*dim=*/-1, /*descending=*/true);
    at::Tensor sp = std::get<0>(sorted);
    at::Tensor si = std::get<1>(sorted);
    at::Tensor keep = at::ones({sp.size(0)}, at::TensorOptions().dtype(at::kBool).device(sp.device()));
    if (use_top_k) {
        at::Tensor rank = at::arange(sp.size(0), sp.options().dtype(at::kLong));
        keep = keep & (rank < params.top_k);
    }
    if (use_top_p) {
        // Nucleus: keep a token when the cumulative mass strictly before it is
        // <= top_p (the crossing token is retained; the top token is always
        // kept, its exclusive prefix being 0).
        at::Tensor excl = at::cumsum(sp, -1) - sp;
        keep = keep & (excl <= params.top_p);
    }
    sp = sp.masked_select(keep);
    si = si.masked_select(keep);
    // The surviving mass no longer sums to 1; multinomial normalizes internally.
    int64_t choice = at::multinomial(sp, 1, /*replacement=*/false, gen).item<int64_t>();
    return si[choice].item<int64_t>();
}

at::Generator make_generator(at::Device device, int64_t seed) {
    at::Generator g = device.is_cuda() ? at::make_generator<at::CUDAGeneratorImpl>(device.index())
                                       : at::make_generator<at::CPUGeneratorImpl>();
    if (seed >= 0) {
        g.set_current_seed(static_cast<uint64_t>(seed));
    } else {
        g.seed();  // draw a nondeterministic seed (unseeded request)
    }
    return g;
}

}  // namespace pulsar
