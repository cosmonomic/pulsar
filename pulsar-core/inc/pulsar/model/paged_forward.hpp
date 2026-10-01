#pragma once

#include "pulsar/model/interface.hpp"  // GroupBatch, StepBatch, StepLogits

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <functional>
#include <utility>

// Model-agnostic primitives of the paged transformer forward, shared by every
// native `*Model`. The per-layer arithmetic (norm placement, projection bias, MLP
// shape, lm_head tying) stays model-specific in the concrete `*Model`.

namespace pulsar {

struct ActiveBuffer;  // defined in active_buffer.hpp

// Apply RoPE (rotate_half over head_dim) to a group's q and k. Inputs are the
// projection outputs, [ntok, n_q_heads * head_dim] and [ntok, n_kv_heads *
// head_dim]; returns them roped and reshaped to [ntok, n_q_heads, head_dim] /
// [ntok, n_kv_heads, head_dim], ready for write_kv + paged attention.
std::pair<at::Tensor, at::Tensor> rope_qk(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& pos,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    double theta
);

// write_kv the group's new K/V into layer `layer`'s pool, then run the group's
// paged attention (prefill vs decode by g.is_prefill()) and return o
// [ntok, n_q_heads, head_dim]. q/k/v are already projected and (q,k) roped in the
// group's [ntok, heads, head_dim] layout.
// lse_capture, when defined, receives this group's per-(query row, query head)
// logsumexp -- the softmax denominator the attention used. fp32 [ntok, n_q_heads].
at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    ActiveBuffer& kv,
    int64_t layer,
    const GroupBatch& g,
    double scale,
    const at::Tensor& lse_capture = {}
);

// Forward skeleton for one group: embed lookup -> n_layers decoder blocks ->
// head. layer_fn(x, layer) is the model-specific decoder block; head_fn(x) is the
// model-specific final norm + lm_head.
at::Tensor paged_group_forward(
    const at::Tensor& embed,
    const GroupBatch& g,
    int64_t n_layers,
    const std::function<at::Tensor(at::Tensor, int64_t)>& layer_fn,
    const std::function<at::Tensor(at::Tensor)>& head_fn
);

// Dispatch a step's present groups through group_fn, packing per-group StepLogits.
// group_fn is the model's group forward.
StepLogits paged_forward(const StepBatch& batch, const std::function<at::Tensor(const GroupBatch&)>& group_fn);

}  // namespace pulsar
