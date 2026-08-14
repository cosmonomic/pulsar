#pragma once

#include "pulsar/model/interface.hpp"

#include <ATen/core/Tensor.h>

#include <ATen/core/Dict.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Qwen3 dense transformer forward in C++, the sibling of Qwen2Model over the same
// paged CUDA ops. Qwen3 differs from Qwen2.5 in two per-layer deltas:
//   - No QKV bias: q/k/v projections are bias-free.
//   - QK-Norm: per-head RMSNorm over head_dim on Q and K, applied before RoPE
//     (weights self_attn.q_norm.weight / k_norm.weight, each [head_dim]).
// head_dim is explicit in the config (may differ from hidden/n_heads); RoPE theta
// comes from config. Everything else (GQA, SwiGLU MLP, pre-norm RMSNorm, final
// norm, tied lm_head, paged-attention ops) matches Qwen2Model.
//
// Engine-internal plain C++, not TorchBind-exposed.

namespace pulsar {

struct ActiveBuffer;  // defined in active_buffer.hpp

struct Qwen3Model : Model {
    // Config scalars + a named-weight dict (HF/Qwen3 naming) already on the pool's
    // device/dtype. The ctor resolves every name to a tensor once into a per-layer
    // struct. lm_head is the untied lm_head.weight when present, else tied to
    // model.embed_tokens.weight.
    Qwen3Model(
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
    );

    StepLogits forward(const StepBatch& batch, ActiveBuffer& kv) override;

    bool supports_capture() const override {
        return true;
    }
    void set_capture_layers(const std::vector<int64_t>& layers) override;
    at::Tensor captured_q(bool is_prefill, int64_t layer) const override;
    at::Tensor captured_lse(bool is_prefill, int64_t layer) const override;

  private:
    // One decoder layer's tensors, resolved from the weight dict at construction.
    // No QKV biases; q_norm/k_norm are the QK-Norm weights.
    struct LayerWeights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor q_proj_weight;
        at::Tensor k_proj_weight;
        at::Tensor v_proj_weight;
        at::Tensor o_proj_weight;
        at::Tensor q_norm_weight;  // [head_dim] per-head RMSNorm on Q
        at::Tensor k_norm_weight;  // [head_dim] per-head RMSNorm on K
        at::Tensor gate_proj_weight;
        at::Tensor up_proj_weight;
        at::Tensor down_proj_weight;
    };

    at::Tensor block(at::Tensor x, int64_t layer, const GroupBatch& g, ActiveBuffer& kv);
    at::Tensor head(at::Tensor x);

    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t hidden;
    int64_t intermediate;
    int64_t vocab;
    double rope_theta;
    double rms_eps;
    double scale;  // 1 / sqrt(head_dim)
    at::Tensor embed_tokens;  // [vocab, hidden]
    std::optional<at::Tensor> lm_head;  // untied head weight, else tie embed_tokens
    at::Tensor norm_weight;  // final RMSNorm
    std::vector<LayerWeights> layers;

    // Pre-RoPE Q capture (QK-attention recall). capture_layers is the set to stash;
    // cap_prefill/cap_decode hold the current forward's per-layer pre-RoPE Q
    // [ntok, n_heads, head_dim] per group, refilled every forward.
    std::unordered_set<int64_t> capture_layers;
    std::unordered_map<int64_t, at::Tensor> cap_prefill;
    std::unordered_map<int64_t, at::Tensor> cap_decode;
    // Softmax denominator of the same forward, same layers: fp32 [ntok, n_heads].
    std::unordered_map<int64_t, at::Tensor> lse_prefill;
    std::unordered_map<int64_t, at::Tensor> lse_decode;
};

}  // namespace pulsar
