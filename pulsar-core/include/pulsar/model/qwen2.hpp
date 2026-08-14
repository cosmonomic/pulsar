#pragma once

#include "pulsar/model/interface.hpp"

#include <ATen/core/Tensor.h>

#include <ATen/core/Dict.h>

#include <cstdint>
#include <string>
#include <vector>

// Qwen2.5 transformer forward in C++ over the pulsar CUDA ops: the native (non-.pt2)
// counterpart of CompiledModel.
//
// The model-agnostic parts (RoPE, write_kv + attention dispatch, the embed ->
// per-layer -> head skeleton, the prefill/decode two-pass) live in paged_forward.hpp.
// Qwen2-specific here: pre/post RMSNorm, QKV projection with bias, SwiGLU MLP
// (gate/up/silu/down, no bias), tied-embedding lm_head with logits in the weight
// dtype (fp32 or bf16).
//
// Engine-internal plain C++, not TorchBind-exposed.

namespace pulsar {

struct ActiveBuffer;  // defined in active_buffer.hpp

struct Qwen2Model : Model {
    // Config scalars + a named-weight dict (HF/Qwen2 naming) already on the pool's
    // device/dtype. The ctor resolves every name to a tensor once into a per-layer
    // struct. lm_head is tied to model.embed_tokens.weight.
    Qwen2Model(
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

    // Run the present groups over the pools `kv`; per-group logits in the weight
    // dtype (fp32 or bf16).
    StepLogits forward(const StepBatch& batch, ActiveBuffer& kv) override;

  private:
    // One decoder layer's tensors, resolved from the weight dict at construction.
    // at::Tensor is an intrusive_ptr handle: these alias the dict's tensors.
    struct LayerWeights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor q_proj_weight;
        at::Tensor q_proj_bias;
        at::Tensor k_proj_weight;
        at::Tensor k_proj_bias;
        at::Tensor v_proj_weight;
        at::Tensor v_proj_bias;
        at::Tensor o_proj_weight;
        at::Tensor gate_proj_weight;
        at::Tensor up_proj_weight;
        at::Tensor down_proj_weight;
    };

    // One Qwen2 decoder block over layer `layer`'s paged KV pool.
    at::Tensor block(at::Tensor x, int64_t layer, const GroupBatch& g, ActiveBuffer& kv);
    // Final RMSNorm + tied-embedding lm_head; logits in the weight dtype.
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
    at::Tensor embed_tokens;  // [vocab, hidden]; tied lm_head
    at::Tensor norm_weight;  // final RMSNorm
    std::vector<LayerWeights> layers;
};

}  // namespace pulsar
