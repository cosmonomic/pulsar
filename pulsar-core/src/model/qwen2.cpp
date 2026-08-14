#include "pulsar/model/qwen2.hpp"

#include "pulsar/runtime/kv/active_buffer.hpp"  // ActiveBuffer (paged KV pools)
#include "pulsar/ops.hpp"  // rmsnorm_cuda
#include "pulsar/model/paged_forward.hpp"  // rope_qk, paged_attention, paged_group_forward

#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace pulsar {

namespace {

at::Tensor need(const c10::Dict<std::string, at::Tensor>& w, const std::string& name) {
    TORCH_CHECK(w.contains(name), "Qwen2Model: missing weight '", name, "'");
    return w.at(name);
}

}  // namespace

Qwen2Model::Qwen2Model(
    int64_t n_layers,
    int64_t n_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    int64_t hidden,
    int64_t intermediate,
    int64_t vocab,
    double rope_theta,
    double rms_eps,
    c10::Dict<std::string, at::Tensor> weights
)
    : n_layers(n_layers),
      n_heads(n_heads),
      n_kv_heads(n_kv_heads),
      head_dim(head_dim),
      hidden(hidden),
      intermediate(intermediate),
      vocab(vocab),
      rope_theta(rope_theta),
      rms_eps(rms_eps),
      scale(1.0 / std::sqrt(static_cast<double>(head_dim))),
      embed_tokens(need(weights, "model.embed_tokens.weight")),
      norm_weight(need(weights, "model.norm.weight")) {
    this->layers.reserve(n_layers);
    for (int64_t i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        this->layers.push_back(
            LayerWeights{
                need(weights, p + "input_layernorm.weight"),
                need(weights, p + "post_attention_layernorm.weight"),
                need(weights, p + "self_attn.q_proj.weight"),
                need(weights, p + "self_attn.q_proj.bias"),
                need(weights, p + "self_attn.k_proj.weight"),
                need(weights, p + "self_attn.k_proj.bias"),
                need(weights, p + "self_attn.v_proj.weight"),
                need(weights, p + "self_attn.v_proj.bias"),
                need(weights, p + "self_attn.o_proj.weight"),
                need(weights, p + "mlp.gate_proj.weight"),
                need(weights, p + "mlp.up_proj.weight"),
                need(weights, p + "mlp.down_proj.weight")
            }
        );
    }
}

// Qwen2 block: QKV projection with bias, RoPE + paged attention, SwiGLU MLP.
at::Tensor Qwen2Model::block(at::Tensor x, int64_t layer, const GroupBatch& g, ActiveBuffer& kv) {
    const LayerWeights& lw = this->layers[layer];
    const int64_t ntok = x.size(0);

    auto h = rmsnorm_cuda(x, lw.input_layernorm, this->rms_eps);
    auto q = at::linear(h, lw.q_proj_weight, lw.q_proj_bias);
    auto k = at::linear(h, lw.k_proj_weight, lw.k_proj_bias);
    auto v = at::linear(h, lw.v_proj_weight, lw.v_proj_bias);

    std::tie(q, k) = rope_qk(q, k, g.pos, this->n_heads, this->n_kv_heads, this->head_dim, this->rope_theta);
    // V is never roped; reshape to the paged ops' [tokens, n_kv_heads, head_dim].
    v = v.view({ntok, this->n_kv_heads, this->head_dim}).contiguous();
    auto o = paged_attention(q, k, v, kv, layer, g, this->scale);

    o = o.reshape({ntok, this->n_heads * this->head_dim});
    x = x + at::linear(o, lw.o_proj_weight);

    auto h2 = rmsnorm_cuda(x, lw.post_attention_layernorm, this->rms_eps);
    auto gate = at::linear(h2, lw.gate_proj_weight);
    auto up = at::linear(h2, lw.up_proj_weight);
    x = x + at::linear(at::silu(gate) * up, lw.down_proj_weight);
    return x;
}

at::Tensor Qwen2Model::head(at::Tensor x) {
    x = rmsnorm_cuda(x, this->norm_weight, this->rms_eps);
    // Tied lm_head; logits come out in the embedding weight's dtype.
    return at::matmul(x, this->embed_tokens.t());
}

StepLogits Qwen2Model::forward(const StepBatch& batch, ActiveBuffer& kv) {
    return paged_forward(batch, [&](const GroupBatch& g) {
        return paged_group_forward(
            this->embed_tokens,
            g,
            this->n_layers,
            [&](at::Tensor x, int64_t layer) { return this->block(x, layer, g, kv); },
            [&](at::Tensor x) { return this->head(x); }
        );
    });
}

}  // namespace pulsar
