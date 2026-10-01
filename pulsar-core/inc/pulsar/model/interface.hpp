#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

// The forward boundary the Engine drives: a scheduler step's prefill and decode
// groups plus the paged KV pools go in, per-group logits come out. The engine
// depends only on this interface; the concrete model (CompiledModel, Qwen2Model)
// is opaque to it.

namespace pulsar {

struct ActiveBuffer;  // owns the paged KV pools; defined in active_buffer.hpp

// One scheduler group's query tokens + KV placement + attention descriptor.
// tokens/pos are the group's tokens and their pos,
// concatenated across the group's sequences; slot_mapping/page_tables place
// them into the pools. The group kind is distinguished by which descriptor is
// defined: a prefill group carries cu_seqlens_q/seqlens_k, a decode group
// carries context_lens. The other descriptor(s) are left undefined.
struct GroupBatch {
    at::Tensor tokens;
    at::Tensor pos;
    at::Tensor slot_mapping;
    at::Tensor page_tables;
    at::Tensor cu_seqlens_q;  // int32 [num_seqs + 1], defined iff prefill
    at::Tensor seqlens_k;  // int32 [num_seqs],     defined iff prefill
    at::Tensor context_lens;  // int32 [num_seqs],     defined iff decode
    // fp32 [num_seqs] in the group's row order: each sequence's attention-mass EMA
    // gain alpha, which the attention kernels index by sequence. Per session, so one
    // batched forward may carry several rates.
    at::Tensor attention_mass_decay;
    // TOKENS: the length the accumulated mass is stated against. <= 0 states it
    // against each query's own attended key count, which only the attention op can
    // read. Engine-level, so one value serves every sequence in the group.
    double mass_length_gain = 0.0;

    bool is_prefill() const {
        return this->cu_seqlens_q.defined();
    }
};

// A scheduler step's two groups; either may be absent.
struct StepBatch {
    std::optional<GroupBatch> prefill;
    std::optional<GroupBatch> decode;
};

// Per-group logits [ntok, vocab] (in the model dtype) for the present groups.
struct StepLogits {
    std::optional<at::Tensor> prefill;
    std::optional<at::Tensor> decode;
};

// tokens + KV + batch -> logits.
struct Model {
    virtual StepLogits forward(const StepBatch& batch, ActiveBuffer& kv) = 0;

    // Pre-RoPE query capture for QK-attention recall. set_capture_layers names the
    // attention layers whose pre-RoPE Q the next forward should stash; after a
    // forward, captured_q(is_prefill, layer) returns that group's pre-RoPE Q
    // [ntok, n_heads, head_dim] in the group's token order (an undefined tensor
    // when the layer was not captured or the group was absent this step). Default:
    // no capture (Q discarded).
    virtual bool supports_capture() const {
        return false;
    }
    virtual void set_capture_layers(const std::vector<int64_t>& layers) {}
    virtual at::Tensor captured_q(bool is_prefill, int64_t layer) const {
        return {};
    }
    // That group's per-(query row, query head) softmax denominator for the same
    // layer, fp32 [ntok, n_heads]; undefined when the layer was not captured.
    virtual at::Tensor captured_lse(bool is_prefill, int64_t layer) const {
        return {};
    }

    virtual ~Model() = default;
};

}  // namespace pulsar
