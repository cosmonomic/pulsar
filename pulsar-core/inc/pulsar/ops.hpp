#pragma once

#include <ATen/core/Tensor.h>

#include <optional>
#include <tuple>

// Kernel entry points. Each is registered as a torch custom op in
// pulsar/ext/ops.cpp, callable from C++ and from Python.

namespace pulsar {

// RMSNorm. Stateless.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// Scatter new K/V rows into a paged pool. k_pool/v_pool are
// [num_pages, page_size, n_kv_heads, head_dim]; k_new/v_new are
// [num_new_tokens, n_kv_heads, head_dim]. Token t writes into
// k_pool[slot/page_size, slot%page_size, :, :] with slot = slot_mapping[t]
// (int32 [num_new_tokens]). Mutates k_pool/v_pool in place. Keys are stored as given:
// the attention ops expect them UNROTATED.
void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& slot_mapping
);

// Paged attention over a batch of sequences, each a VIEW: an ordered list of lines
// (page_size-key pages) of the shared pool. Key index j of sequence s lives at
// (page_tables[s][j / page_size], j % page_size). A line may appear in several views.
//
// The pool holds UNROTATED keys. The op rotates key j of sequence s to the RoPE
// position rope_layout[s] gives index j, in the HF/Qwen2 rotate_half convention over
// the full head_dim with base rope_theta. rope_layout is int32 [num_seqs, 3] holding
// {n_sink, working_lo, short_offset}:
//   j < n_sink                 -> j
//   n_sink <= j < working_lo   -> short_offset
//   j >= working_lo            -> short_offset + 1 + j - working_lo
// so {0, 0, -1} is the identity and a compacted layout collapses the distant region
// [n_sink, working_lo) onto short_offset. Queries arrive ALREADY rotated, by the
// caller, to the same layout's position of their own view index.
//
// mass, when given, receives each attended key's normalized softmax weight, added into
// the QUERY head's own column (no sum over the group), at the key's VIEW row:
// sequence s's key j accumulates into mass[cu_view_pages[s] + j / page_size,
// j % page_size, h]. Views therefore never share mass, whatever lines they share.
//   mass                 fp32 [total_view_pages, page_size, n_q_heads], mutated in place
//   cu_view_pages        int32 [num_seqs + 1]; sequence s owns view pages
//                        [cu_view_pages[s], cu_view_pages[s+1]), at least its context's
//   attention_mass_decay fp32 [num_seqs]; sequence s's EMA gain alpha, the factor its
//                        received weight is scaled by (1.0 accumulates the raw weight)
//   mass_length_gain     TOKENS. Each query's weight is rescaled by the length its
//                        share is stated against, so uniform attention over exactly
//                        that length stores 1. <= 0 takes the keys that query attended,
//                        which only the op can read. A caller wanting another CONSTANT
//                        length passes 1 and multiplies once afterwards: the mass sums
//                        over query tokens, so a constant commutes with it.
// The three tensors are passed together or not at all; without them the op skips its
// mass pass entirely.
//
// lse_capture, when given, receives each (query row, query head)'s logsumexp of
// scale * q.k over the keys that row attended, fp32 contiguous [total_q, n_q_heads]:
// the softmax denominator the attention used, so exp(logit - lse) is a key's weight.

// Decode: one query per sequence (q [num_seqs, n_q_heads, head_dim]) attending its
// whole view, keys [0, context_lens[s]). num_splits > 0 forces the split (flash-decode)
// path with that many splits, 1 included; 0 picks the split count from occupancy.
// Returns o [num_seqs, n_q_heads, head_dim].
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& rope_layout,
    double rope_theta,
    double scale,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt,
    int64_t num_splits = 0
);

// attn_decode on the scalar kernel alone: the reference the tensor-core path is
// checked against.
at::Tensor attn_decode_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& rope_layout,
    double rope_theta,
    double scale,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt
);

// Prefill: a ragged batch of query chunks, causal over each view. Sequence i's queries
// are q[cu_seqlens_q[i] : cu_seqlens_q[i+1]] (q [total_q, n_q_heads, head_dim]) at view
// indices [ctx_start_i, seqlens_k[i]) with ctx_start_i = seqlens_k[i] - seq_q_i, and
// the query at view index p attends keys [0, p]. The mass is the within-chunk EMA: the
// query at offset o from its chunk's end contributes alpha * (1 - alpha)^o times its
// weight. Returns o [total_q, n_q_heads, head_dim].
at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& rope_layout,
    double rope_theta,
    double scale,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt
);

// attn_prefill on the scalar kernel alone: the reference the tensor-core path is
// checked against.
at::Tensor attn_prefill_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& rope_layout,
    double rope_theta,
    double scale,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt
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
