#include "pulsar/model/paged_forward.hpp"

#include "pulsar/runtime/kv/active_buffer.hpp"  // ActiveBuffer (paged KV pools)
#include "pulsar/ops.hpp"  // rope_cuda, write_kv_cuda, attn_prefill/decode_cuda

#include <ATen/ATen.h>

namespace pulsar {

std::pair<at::Tensor, at::Tensor> rope_qk(
    const at::Tensor& q_in,
    const at::Tensor& k_in,
    const at::Tensor& pos,
    int64_t n_q_heads,
    int64_t n_kv_heads,
    int64_t head_dim,
    double theta
) {
    const int64_t ntok = q_in.size(0);
    // rope_cuda requires [heads, seq, head_dim].
    auto q = q_in.view({ntok, n_q_heads, head_dim}).transpose(0, 1).contiguous();
    auto k = k_in.view({ntok, n_kv_heads, head_dim}).transpose(0, 1).contiguous();
    q = rope_cuda(q, pos, theta);
    k = rope_cuda(k, pos, theta);
    q = q.transpose(0, 1).contiguous();  // transpose to paged layout [ntok, n_q_heads, head_dim]
    k = k.transpose(0, 1).contiguous();  // transpose to paged layout [ntok, n_kv_heads, head_dim]
    return {q, k};
}

at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    ActiveBuffer& kv,
    int64_t layer,
    const GroupBatch& g,
    double scale,
    const at::Tensor& lse_capture
) {
    auto k_pool = kv.k_pool(layer);
    auto v_pool = kv.v_pool(layer);
    auto mass_pool = kv.mass_pool(layer);
    // Prefill pays a whole second streaming pass for mass, so a layer nothing reduces
    // over hands the op an undefined pool and skips it. Decode's mass rides in its
    // single pass and stays unconditional.
    auto prefill_mass = kv.mass_is_read(layer) ? mass_pool : at::Tensor{};
    write_kv_cuda(k_pool, v_pool, k, v, g.slot_mapping);
    // The attention-mass EMA gain alpha is per SEQUENCE (g.attention_mass_decay, one
    // entry per group row); the prefill within-chunk grading derives its retention
    // 1 - alpha from the same entry.
    if (g.is_prefill()) {
        return attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            prefill_mass,
            g.page_tables,
            g.cu_seqlens_q,
            g.seqlens_k,
            scale,
            g.attention_mass_decay,
            lse_capture.defined() ? std::optional<at::Tensor>(lse_capture) : std::nullopt,
            g.mass_length_gain
        );
    }
    return attn_decode_cuda(
        q,
        k_pool,
        v_pool,
        mass_pool,
        g.page_tables,
        g.context_lens,
        scale,
        g.attention_mass_decay,
        lse_capture.defined() ? std::optional<at::Tensor>(lse_capture) : std::nullopt,
        g.mass_length_gain
    );
}

at::Tensor paged_group_forward(
    const at::Tensor& embed,
    const GroupBatch& g,
    int64_t n_layers,
    const std::function<at::Tensor(at::Tensor, int64_t)>& layer_fn,
    const std::function<at::Tensor(at::Tensor)>& head_fn
) {
    auto x = embed.index_select(0, g.tokens);  // [ntok, hidden]
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        x = layer_fn(x, layer);
    }
    return head_fn(x);
}

StepLogits paged_forward(const StepBatch& batch, const std::function<at::Tensor(const GroupBatch&)>& group_fn) {
    StepLogits out;
    if (batch.prefill.has_value()) {
        out.prefill = group_fn(*batch.prefill);
    }
    if (batch.decode.has_value()) {
        out.decode = group_fn(*batch.decode);
    }
    return out;
}

}  // namespace pulsar
