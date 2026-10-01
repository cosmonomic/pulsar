#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

#include <tuple>

// Kernel entry points. Each is registered as a torch custom op in
// pulsar/ext/ops.cpp, callable from C++ and from Python.

namespace pulsar {

// RMSNorm. Stateless.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// Fused causal attention with a per-key attention_mass side output. attention_mass
// is threaded in and returned with this step's mass added.
//   q          [n_q_heads, seq_q, head_dim]
//   k, v       [n_kv_heads, L, head_dim]
//   attention_mass [n_kv_heads, L]  (fp32; running signal, added to)
// Returns (o [n_q_heads, seq_q, head_dim], attention_mass_out [n_kv_heads, L]).
std::tuple<at::Tensor, at::Tensor> attn_causal_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& attention_mass,
    double scale,
    int64_t causal_offset
);

// Cache-form fused causal attention for the fixed-capacity KV path. k_cache and
// v_cache are [n_kv_heads, max_len, head_dim] fixed buffers whose new K/V rows
// are already written at [cur_len, cur_len+seq_q); only rows [0, valid_len) with
// valid_len = cur_len + seq_q are read. attention_mass [n_kv_heads, max_len] fp32
// is mutated in place (this step's per-key normalized mass is added into rows
// [0, valid_len)). Returns o [n_q_heads, seq_q, head_dim].
at::Tensor attn_causal_cache_cuda(
    const at::Tensor& q,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& attention_mass,
    double scale,
    int64_t cur_len
);

// Scatter new K/V rows into a paged pool. k_pool/v_pool are
// [num_pages, page_size, n_kv_heads, head_dim]; k_new/v_new are
// [num_new_tokens, n_kv_heads, head_dim]. Token t writes into
// k_pool[slot/page_size, slot%page_size, :, :] with slot = slot_mapping[t]
// (int32 [num_new_tokens]). Mutates k_pool/v_pool in place.
void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& slot_mapping
);

// Paged decode attention with a per-key mass side output (seq_q == 1). For
// sequence s and query head h (kv head h / (n_q_heads / n_kv_heads)), attend
// over keys j in [0, context_lens[s]); key j lives at pool slot
// (page_tables[s][j/page_size], j%page_size). Non-causal (the decode query
// attends its whole context). The normalized weight scaled by the EMA gain alpha
// is added into mass_pool[page, offset, h] via atomicAdd, into the QUERY head's
// own column with no sum over the group. mass_pool
// [num_pages, page_size, n_q_heads] fp32 is mutated in place. attention_mass_decay
// is fp32 [num_seqs] on the pool device: sequence s's EMA gain alpha, the factor
// its accumulated per-key term is scaled by (1.0 accumulates the raw weight). One
// alpha per sequence, so sequences batched together may decay at different rates.
// The accumulated weight is rescaled by mass_length_gain, the length its share is
// stated against, so uniform attention over exactly that length stores 1 and the stored
// quantity is the key's share of attention as a multiple of UNIFORM AT THAT LENGTH.
// <= 0 takes context_lens[s], the keys the query actually attended, which the op can
// read but a caller cannot. A caller wanting some other CONSTANT length passes 1 and
// multiplies by it once afterwards: the mass is a sum over query tokens, so a constant
// commutes with the accumulation and the op need not carry it.
//   q            [num_seqs, n_q_heads, head_dim]  (already RoPE'd)
//   k_pool,v_pool [num_pages, page_size, n_kv_heads, head_dim]
//   page_tables int32 [num_seqs, max_pages]
//   context_lens int32 [num_seqs]
//   attention_mass_decay fp32 [num_seqs]
//   mass_length_gain  TOKENS; <= 0 uses each sequence's own context length
// Returns o [num_seqs, n_q_heads, head_dim].
// lse_capture, when defined, receives this call's per-(query row, query head)
// logsumexp of scale * q.k over the keys the row attended: fp32 contiguous
// [num_rows, n_q_heads]. It is the softmax denominator the attention itself used, so
// exp(logit - lse) is that key's attention weight. Undefined => not written.
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture = std::nullopt,
    double mass_length_gain = 0.0
);

// Scalar-only reference sibling of attn_decode: same semantics and result, always
// the scalar kernel (never the tensor-core path).
at::Tensor attn_decode_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain = 0.0
);

// Split-K (flash-decode) variant of attn_decode with the split count forced to
// num_splits (>= 1). Same semantics and result. num_splits is consulted only on the
// tensor-core path; the scalar fallback ignores it.
at::Tensor attn_decode_split_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    double scale,
    int64_t num_splits,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture = std::nullopt,
    double mass_length_gain = 0.0
);

// Paged prefill attention with a per-key mass side output (seq_q >= 1, causal,
// varlen). The prefill counterpart of attn_decode: for a
// ragged batch, sequence i's query tokens are q[cu_seqlens_q[i] :
// cu_seqlens_q[i+1]] and occupy context pos [ctx_start_i, seqlens_k[i])
// with ctx_start_i = seqlens_k[i] - seq_q_i. A query token at context pos p
// attends to keys j in [0, p] (causal); key j lives at pool slot
// (page_tables[i][j/page_size], j%page_size). The normalized weight is added
// into mass_pool at each attended key's slot via atomicAdd, into the QUERY head's
// own column with no sum over the group, summed over the query tokens with
// p >= j. attention_mass_decay is fp32 [num_seqs] on the pool device: sequence i's
// EMA gain alpha, the factor its accumulated per-key term is scaled by (1.0
// accumulates the raw weight). The within-chunk retention 1 - alpha comes from the
// same value: a query at offset o from the chunk end weighs (1 - alpha)^o. One
// alpha per sequence, so sequences batched together may decay at different rates.
// Each query's contribution is rescaled by mass_length_gain, the length its share is
// stated against. <= 0 takes p + 1, that query's own CAUSAL key count, which varies
// query by query: mass handed out early in a prefill (few keys attended, large
// weights) is then comparable with mass handed out later, and the stored quantity is
// the key's share of attention as a multiple of UNIFORM. A caller wanting some other
// CONSTANT length passes 1 and multiplies by it once afterwards (see attn_decode).
//   q            [total_q, n_q_heads, head_dim]  (already RoPE'd)
//   k_pool,v_pool [num_pages, page_size, n_kv_heads, head_dim]
//   mass_pool    [num_pages, page_size, n_q_heads]  fp32, per query head; UNDEFINED
//                skips the mass pass entirely, and nothing else in the op reads it
//   page_tables int32 [num_seqs, max_pages]
//   cu_seqlens_q int32 [num_seqs+1]
//   seqlens_k    int32 [num_seqs]
//   attention_mass_decay fp32 [num_seqs]
//   mass_length_gain  TOKENS; <= 0 uses each query's own causal key count
// Returns o [total_q, n_q_heads, head_dim].
// lse_capture, when defined, receives the per-(query row, query head) logsumexp over
// the keys that row attended -- its CAUSAL prefix, so the key count varies by row --
// as fp32 contiguous [total_q, n_q_heads]. Undefined => not written.
at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    double scale,
    const at::Tensor& attention_mass_decay,
    const std::optional<at::Tensor>& lse_capture = std::nullopt,
    double mass_length_gain = 0.0
);

// Scalar-only reference sibling of attn_prefill: same semantics and result, always
// the scalar kernel (never the tensor-core path).
at::Tensor attn_prefill_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& mass_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    double scale,
    const at::Tensor& attention_mass_decay,
    double mass_length_gain = 0.0
);

// Fused w4a16 dequant-GEMM: y = x @ dequant(W)^T for a Linear with group-wise
// symmetric int4 weights. Computes y[m,n] = sum_k x[m,k] * w_deq[n,k] with
// w_deq[n,k] = int4(weight_packed[n,k]) * weight_scale[n, k/group_size].
//   x             [M, K]    fp16/bf16, contiguous
//   weight_packed [N, K/8]  int32, 8 signed int4 per int32 along K (LSB-first)
//   weight_scale  [N, K/group_size]  per-group scale (upcast to fp32 internally)
//   group_size    quant group along K (multiple of 16; K a multiple of it)
// Returns y [M, N], same dtype as x.
at::Tensor gemm_w4a16_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size
);

// Split-K variant of gemm_w4a16 with the K-split count forced to num_splits
// (>= 1). Same semantics and result; always routes through the split+combine path
// (even at num_splits == 1; num_splits may exceed the K-chunk count, the extra
// splits being empty and contributing 0).
at::Tensor gemm_w4a16_split_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int64_t num_splits
);

// Marlin-style w4a16 dequant-GEMM: one-time weight repack into a fragment-matched
// int4 layout + an online GEMM consuming it. Same math as gemm_w4a16.
//   repack_w4a16(weight_packed [N,K/8] int32, weight_scale [N,K/G], group_size)
//     -> (weight_marlin int32 [Npad*Kpad/8], scale_marlin fp32 [K/G, N])
//        (opaque layout; Npad/Kpad are N/K rounded up to the Marlin tiling).
//   gemm_w4a16_marlin(x [M,K], weight_marlin, scale_marlin, group_size) -> y [M,N]
std::tuple<at::Tensor, at::Tensor>
repack_w4a16_cuda(const at::Tensor& weight_packed, const at::Tensor& weight_scale, int64_t group_size);

at::Tensor gemm_w4a16_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t group_size
);

// Rotary position embedding (RoPE), HF/Qwen2 "rotate_half" convention over the
// full head_dim. Stateless.
//   x         [n_heads, seq, head_dim]  (head_dim even)
//   pos [seq]  int64; pos of each row (may be arbitrary / non-contiguous).
//   theta     RoPE base (e.g. 1000000 for Qwen2.5, 10000 default).
// Returns the rotated x, same shape and dtype.
at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, double theta);

// Reposition already-RoPE'd keys in a paged KV pool by rotating each by its delta
// angle:  reposition(rope(raw, p_old), p_old -> p_new) == rope(raw, p_new).
// Same HF/Qwen2 rotate_half convention as rope_cuda; K only (V is untouched).
//   k_pool        [num_pages, page_size, n_kv_heads, head_dim]  (ONE layer;
//                 mutated in place, contiguous)
//   slots         int32 [M]; physical slot of each key in the flattened
//                 [num_pages*page_size] first two dims.
//   old_positions int32 [M]; the pos each key is currently roped at.
//   new_positions int32 [M]; the pos to rotate each key to.
//   theta         RoPE base (must match the base used to rope the keys).
void reposition_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& slots,
    const at::Tensor& old_positions,
    const at::Tensor& new_positions,
    double theta
);

}  // namespace pulsar
