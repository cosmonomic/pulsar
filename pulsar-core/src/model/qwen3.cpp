#include "pulsar/model/qwen3.hpp"

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
    TORCH_CHECK(w.contains(name), "Qwen3Model: missing weight '", name, "'");
    return w.at(name);
}

}  // namespace

Qwen3Model::Qwen3Model(
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
    // Untied lm_head when the checkpoint ships one, else tied to the token embedding.
    if (weights.contains("lm_head.weight")) {
        this->lm_head = weights.at("lm_head.weight");
    }
    this->layers.reserve(n_layers);
    for (int64_t i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        this->layers.push_back(
            LayerWeights{
                need(weights, p + "input_layernorm.weight"),
                need(weights, p + "post_attention_layernorm.weight"),
                need(weights, p + "self_attn.q_proj.weight"),
                need(weights, p + "self_attn.k_proj.weight"),
                need(weights, p + "self_attn.v_proj.weight"),
                need(weights, p + "self_attn.o_proj.weight"),
                need(weights, p + "self_attn.q_norm.weight"),
                need(weights, p + "self_attn.k_norm.weight"),
                need(weights, p + "mlp.gate_proj.weight"),
                need(weights, p + "mlp.up_proj.weight"),
                need(weights, p + "mlp.down_proj.weight")
            }
        );
    }
}

// Qwen3 block: bias-free QKV projection, per-head QK-Norm before RoPE, then RoPE +
// paged attention and a SwiGLU MLP.
at::Tensor Qwen3Model::block(at::Tensor x, int64_t layer, const GroupBatch& g, ActiveBuffer& kv) {
    const LayerWeights& lw = this->layers[layer];
    const int64_t ntok = x.size(0);

    auto h = rmsnorm_cuda(x, lw.input_layernorm, this->rms_eps);
    auto q = at::linear(h, lw.q_proj_weight);
    auto k = at::linear(h, lw.k_proj_weight);
    auto v = at::linear(h, lw.v_proj_weight);

    // QK-Norm: per-head RMSNorm over head_dim, before RoPE. Reshape so the last
    // axis is the per-head vector, then flatten back for rope_qk.
    q = rmsnorm_cuda(q.view({ntok, this->n_heads, this->head_dim}), lw.q_norm_weight, this->rms_eps)
            .view({ntok, this->n_heads * this->head_dim});
    k = rmsnorm_cuda(k.view({ntok, this->n_kv_heads, this->head_dim}), lw.k_norm_weight, this->rms_eps)
            .view({ntok, this->n_kv_heads * this->head_dim});

    // Stash the post-QK-Norm, pre-RoPE Q for QK-attention recall.
    // [ntok, n_heads, head_dim] in the group's token order.
    if (this->capture_layers.count(layer)) {
        at::Tensor q_pre = q.view({ntok, this->n_heads, this->head_dim}).detach();
        (g.is_prefill() ? this->cap_prefill : this->cap_decode)[layer] = q_pre;
    }

    std::tie(q, k) = rope_qk(q, k, g.pos, this->n_heads, this->n_kv_heads, this->head_dim, this->rope_theta);
    v = v.view({ntok, this->n_kv_heads, this->head_dim}).contiguous();
    // Capture the softmax denominator on the same layers as Q: the recall score needs
    // the denominator the attention actually used, not one rebuilt over a subset.
    at::Tensor lse;
    if (this->capture_layers.count(layer)) {
        lse = at::empty({ntok, this->n_heads}, at::device(x.device()).dtype(at::kFloat));
    }
    auto o = paged_attention(q, k, v, kv, layer, g, this->scale, lse);
    if (lse.defined()) {
        (g.is_prefill() ? this->lse_prefill : this->lse_decode)[layer] = lse;
    }

    o = o.reshape({ntok, this->n_heads * this->head_dim});
    x = x + at::linear(o, lw.o_proj_weight);

    auto h2 = rmsnorm_cuda(x, lw.post_attention_layernorm, this->rms_eps);
    auto gate = at::linear(h2, lw.gate_proj_weight);
    auto up = at::linear(h2, lw.up_proj_weight);
    x = x + at::linear(at::silu(gate) * up, lw.down_proj_weight);
    return x;
}

at::Tensor Qwen3Model::head(at::Tensor x) {
    x = rmsnorm_cuda(x, this->norm_weight, this->rms_eps);
    // Logits in the weight dtype: untied head weight when present, else tied token
    // embedding.
    const at::Tensor& w = this->lm_head.has_value() ? *this->lm_head : this->embed_tokens;
    return at::matmul(x, w.t());
}

StepLogits Qwen3Model::forward(const StepBatch& batch, ActiveBuffer& kv) {
    // Fresh capture each forward: clear both groups' stashes; block() refills the
    // configured layers.
    this->lse_prefill.clear();
    this->lse_decode.clear();
    this->cap_prefill.clear();
    this->cap_decode.clear();
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

void Qwen3Model::set_capture_layers(const std::vector<int64_t>& layers) {
    this->capture_layers = std::unordered_set<int64_t>(layers.begin(), layers.end());
}

at::Tensor Qwen3Model::captured_q(bool is_prefill, int64_t layer) const {
    const auto& cap = is_prefill ? this->cap_prefill : this->cap_decode;
    auto it = cap.find(layer);
    return it == cap.end() ? at::Tensor{} : it->second;
}

at::Tensor Qwen3Model::captured_lse(bool is_prefill, int64_t layer) const {
    const auto& cap = is_prefill ? this->lse_prefill : this->lse_decode;
    auto it = cap.find(layer);
    return it == cap.end() ? at::Tensor{} : it->second;
}

}  // namespace pulsar
