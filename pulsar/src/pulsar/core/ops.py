"""Typed wrappers over the C++ custom-op kernels, and their meta kernels.

The meta kernels are what torch.export and torch.library.opcheck trace against.
pulsar_core.so is loaded when pulsar.core is imported.
"""

import torch


def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


def _rope_fake(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    return torch.empty_like(x)


def _reposition_kv_fake(
    k_pool: torch.Tensor,
    slots: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    theta: float,
) -> None:
    # k_pool is mutated in place (Tensor(a!)); the op returns None.
    return None


def _write_kv_fake(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    # k_pool/v_pool are mutated in place (Tensor(a!)/Tensor(b!)); returns None.
    return None


def _attn_decode_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
    num_splits: int = 0,
) -> torch.Tensor:
    # mass/lse_capture are mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_decode_scalar_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


def _attn_prefill_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
) -> torch.Tensor:
    # mass/lse_capture are mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_prefill_scalar_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(q)


def _gemm_w4a16_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    n = weight_packed.shape[0]
    return x.new_empty((x.shape[0], n))


def _gemm_w4a16_split_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    num_splits: int,
) -> torch.Tensor:
    n = weight_packed.shape[0]
    return x.new_empty((x.shape[0], n))


# Marlin tiling constants; must match pulsar-core/src/kernels/w4a16_marlin.cu.
_MARLIN_BN = 64
_MARLIN_BK = 64


def _repack_w4a16_fake(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = weight_packed.shape[0]
    k = weight_packed.shape[1] * 8
    npad = -(-n // _MARLIN_BN) * _MARLIN_BN
    kpad = -(-k // _MARLIN_BK) * _MARLIN_BK
    total = npad * kpad // 8
    weight_marlin = weight_packed.new_empty((total,))
    scale_marlin = weight_scale.new_empty((k // group_size, n), dtype=torch.float32)
    return weight_marlin, scale_marlin


def _gemm_w4a16_marlin_fake(
    x: torch.Tensor,
    weight_marlin: torch.Tensor,
    scale_marlin: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    n = scale_marlin.shape[1]
    return x.new_empty((x.shape[0], n))


def _register_fakes() -> None:
    """Register meta/FakeTensor kernels so the ops cross torch.export/AOTInductor.

    Each fake must accept its op's whole schema (pulsar/ext/ops.cpp). The
    dispatcher elides trailing arguments left at their default, so a fake missing a
    trailing parameter only fails once a caller passes one.
    """
    fakes = {
        "pulsar::rmsnorm": _rmsnorm_fake,
        "pulsar::rope": _rope_fake,
        "pulsar::reposition_kv": _reposition_kv_fake,
        "pulsar::write_kv": _write_kv_fake,
        "pulsar::attn_decode": _attn_decode_fake,
        # The _scalar ops are the validation seam (scalar oracle), registered here
        # for tests; pulsar.ops has no public wrapper for them.
        "pulsar::attn_decode_scalar": _attn_decode_scalar_fake,
        "pulsar::attn_prefill": _attn_prefill_fake,
        "pulsar::attn_prefill_scalar": _attn_prefill_scalar_fake,
        "pulsar::gemm_w4a16": _gemm_w4a16_fake,
        "pulsar::gemm_w4a16_split": _gemm_w4a16_split_fake,
        "pulsar::repack_w4a16": _repack_w4a16_fake,
        "pulsar::gemm_w4a16_marlin": _gemm_w4a16_marlin_fake,
    }
    for name, func in fakes.items():
        torch.library.register_fake(name, func)


_register_fakes()


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension."""
    return torch.ops.pulsar.rmsnorm(x, weight, eps)


def write_kv(
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter new K/V rows into a paged KV pool.

    Args:
        k_pool: Key pool, [num_pages, page_size, n_kv_heads, head_dim]. Mutated
            in place.
        v_pool: Value pool, same shape as k_pool. Mutated in place.
        k_new: New keys, [num_new_tokens, n_kv_heads, head_dim], unrotated: the
            attention ops rotate each key to its view position as they read it.
        v_new: New values, same shape as k_new.
        slot_mapping: int32 [num_new_tokens]. Token t is written into
            k_pool[slot/page_size, slot%page_size, :, :] with slot =
            slot_mapping[t] (page_size = k_pool.size(1)).

    Returns:
        None. k_pool and v_pool are updated in place.
    """
    torch.ops.pulsar.write_kv(k_pool, v_pool, k_new, v_new, slot_mapping)


def attn_decode(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
    num_splits: int = 0,
) -> torch.Tensor:
    """Paged decode attention over per-sequence views (seq_q == 1).

    Each sequence contributes one query, already rotated to its position, that
    attends its whole view (non-causal). The op rotates every key to its view
    position as it reads it.

    Args:
        q: Query tokens, [num_seqs, n_q_heads, head_dim].
        k_pool: Key pool, [num_lines, page_size, n_kv_heads, head_dim], UNROTATED.
        v_pool: Value pool, same shape as k_pool.
        page_tables: int32 [num_seqs, max_pages]. page_tables[s][b] is the line
            holding view page b of sequence s; key index j lives at
            (page_tables[s][j // page_size], j % page_size). A line may appear in
            several views.
        context_lens: int32 [num_seqs]. Number of keys in each view.
        rope_layout: int32 [num_seqs, 3], {n_sink, working_lo, short_offset} per
            sequence. Key index j is rotated to j below n_sink, to short_offset below
            working_lo, and to short_offset + 1 + j - working_lo from there on, so
            {0, 0, -1} is the identity.
        rope_theta: RoPE base the keys are rotated with (rotate_half convention).
        scale: Softmax scale; softmax(scale * q @ k^T).
        mass: fp32 [total_view_pages, page_size, n_q_heads], mutated in place.
            Sequence s's key j adds its normalized weight into row
            (cu_view_pages[s] + j // page_size, j % page_size), each query head into
            its own column, so views sharing a line accumulate apart. None skips the
            mass pass; cu_view_pages and attention_mass_decay then go unread.
        cu_view_pages: int32 [num_seqs + 1]. Sequence s owns view pages
            [cu_view_pages[s], cu_view_pages[s+1]), at least as many as its context
            spans.
        attention_mass_decay: fp32 [num_seqs]. Sequence s's EMA gain alpha, the
            factor its per-key mass term is scaled by; 1.0 accumulates the raw
            normalized weight.
        mass_length_gain: TOKENS. The length each query's weight is stated against,
            so uniform attention over exactly that many keys stores 1. 0 or less
            takes the keys that query attended, which only the op can read. For any
            other CONSTANT length, pass 1 and multiply the result by it once
            afterwards.
        lse_capture: fp32 contiguous [num_seqs, n_q_heads], written in place with
            each row's logsumexp of scale * q @ k^T over the keys it attended. None
            leaves it unwritten.
        num_splits: Forces the split (flash-decode) path with that many splits when
            > 0; 0 picks the count from occupancy.

    Returns:
        o, [num_seqs, n_q_heads, head_dim].
    """
    return torch.ops.pulsar.attn_decode(
        q, k_pool, v_pool, page_tables, context_lens, rope_layout, rope_theta, scale,
        mass, cu_view_pages, attention_mass_decay, mass_length_gain, lse_capture,
        num_splits,
    )


def attn_prefill(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    rope_layout: torch.Tensor,
    rope_theta: float,
    scale: float,
    mass: torch.Tensor | None = None,
    cu_view_pages: torch.Tensor | None = None,
    attention_mass_decay: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
    lse_capture: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paged prefill attention over per-sequence views (causal, varlen).

    Sequence i's queries are q[cu_seqlens_q[i] : cu_seqlens_q[i+1]], already rotated,
    at view indices [ctx_start_i, seqlens_k[i]) with ctx_start_i = seqlens_k[i] -
    seq_q_i; the query at view index p attends keys [0, p]. The op rotates every key
    to its view position as it reads it. The mass is the within-chunk EMA: the query
    at offset o from its chunk's end contributes alpha * (1 - alpha)**o times its
    weight.

    Args:
        q: Query tokens, [total_q, n_q_heads, head_dim].
        k_pool: Key pool, [num_lines, page_size, n_kv_heads, head_dim], UNROTATED.
        v_pool: Value pool, same shape as k_pool.
        page_tables: int32 [num_seqs, max_pages]. page_tables[s][b] is the line
            holding view page b of sequence s; key index j lives at
            (page_tables[s][j // page_size], j % page_size). A line may appear in
            several views.
        cu_seqlens_q: int32 [num_seqs+1]. Prefix sums of per-seq query lengths.
        seqlens_k: int32 [num_seqs]. Keys in each view (>= its query length).
        rope_layout: int32 [num_seqs, 3], {n_sink, working_lo, short_offset} per
            sequence. Key index j is rotated to j below n_sink, to short_offset below
            working_lo, and to short_offset + 1 + j - working_lo from there on, so
            {0, 0, -1} is the identity.
        rope_theta: RoPE base the keys are rotated with (rotate_half convention).
        scale: Softmax scale; softmax(scale * q @ k^T).
        mass: fp32 [total_view_pages, page_size, n_q_heads], mutated in place.
            Sequence s's key j adds its normalized weight into row
            (cu_view_pages[s] + j // page_size, j % page_size), each query head into
            its own column, so views sharing a line accumulate apart. None skips the
            mass pass; cu_view_pages and attention_mass_decay then go unread.
        cu_view_pages: int32 [num_seqs + 1]. Sequence s owns view pages
            [cu_view_pages[s], cu_view_pages[s+1]), at least as many as its context
            spans.
        attention_mass_decay: fp32 [num_seqs]. Sequence s's EMA gain alpha, the
            factor its per-key mass term is scaled by; 1.0 accumulates the raw
            normalized weight.
        mass_length_gain: TOKENS. The length each query's weight is stated against,
            so uniform attention over exactly that many keys stores 1. 0 or less
            takes the keys that query attended, which only the op can read. For any
            other CONSTANT length, pass 1 and multiply the result by it once
            afterwards.
        lse_capture: fp32 contiguous [total_q, n_q_heads], written in place with each
            query row's logsumexp of scale * q @ k^T over its causal prefix. None
            leaves it unwritten.

    Returns:
        o, [total_q, n_q_heads, head_dim].
    """
    return torch.ops.pulsar.attn_prefill(
        q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, rope_layout,
        rope_theta, scale, mass, cu_view_pages, attention_mass_decay,
        mass_length_gain, lse_capture,
    )


def gemm_w4a16(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Fused w4a16 dequant-GEMM computing y = x @ dequant(W)^T.

    Computes y[m, n] = sum_k x[m, k] * w_deq[n, k] for a Linear with group-wise
    symmetric int4 weights, w_deq[n, k] = int4(weight_packed[n, k]) *
    weight_scale[n, k // group_size]. The int4 weights are read packed from
    global memory and dequantized only into an on-chip tensor-core tile.

    Args:
        x: Activations, [M, K], fp16 or bf16, contiguous.
        weight_packed: int32 [N, K/8]. Eight signed int4 values per int32 along
            K, LSB-first: column k lives in nibble k % 8 of weight_packed[n,
            k // 8], stored two's-complement in [-8, 7].
        weight_scale: Per-group scale, [N, K/group_size]. Upcast to fp32
            internally.
        group_size: Quantization group size along K (a multiple of 16; K a
            multiple of group_size).

    Returns:
        y, [M, N], same dtype as x.
    """
    return torch.ops.pulsar.gemm_w4a16(x, weight_packed, weight_scale, group_size)


def repack_w4a16(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack simple-format w4a16 weights into the Marlin fragment layout.

    One-time load-step transform of the on-disk compressed-tensors packing into
    the opaque layout gemm_w4a16_marlin consumes. Same quantization math; only
    the byte layout changes.

    Args:
        weight_packed: int32 [N, K/8], the simple LSB-first int4 packing (same as
            gemm_w4a16 expects).
        weight_scale: Per-group scale, [N, K/group_size].
        group_size: Quantization group size along K (a multiple of 16).

    Returns:
        (weight_marlin, scale_marlin): weight_marlin is int32 [Npad*Kpad/8] in the
        opaque Marlin layout (int4, never dequantized here); scale_marlin is fp32
        [K/group_size, N] (the transpose of weight_scale). Npad/Kpad are N/K
        rounded up to the Marlin tiling.
    """
    return torch.ops.pulsar.repack_w4a16(weight_packed, weight_scale, group_size)


def gemm_w4a16_marlin(
    x: torch.Tensor,
    weight_marlin: torch.Tensor,
    scale_marlin: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Marlin-style fused w4a16 dequant-GEMM computing y = x @ dequant(W)^T.

    Same result as gemm_w4a16 but consumes the repacked (weight_marlin,
    scale_marlin) from repack_w4a16. The int4 weights are read from global memory
    and dequantized only into on-chip tensor-core registers (fp32 accumulate).

    Args:
        x: Activations, [M, K], fp16 or bf16, contiguous.
        weight_marlin: Opaque int32 Marlin-layout weights from repack_w4a16.
        scale_marlin: fp32 [K/group_size, N] scales from repack_w4a16.
        group_size: Quantization group size along K (a multiple of 16).

    Returns:
        y, [M, N], same dtype as x.
    """
    return torch.ops.pulsar.gemm_w4a16_marlin(x, weight_marlin, scale_marlin, group_size)


def rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotary position embedding, HF/Qwen2 rotate_half convention.

    Rotates the full head_dim (no partial-rotary fraction). The angle is formed
    in fp64 and the rotation runs in fp32; the result is cast back to x's dtype.

    Args:
        x: Input tokens, [n_heads, seq, head_dim], head_dim even.
        pos: Pos of each row, int64 [seq]. May be arbitrary
            or non-contiguous, so a cache can be re-RoPE'd under a new layout.
        theta: RoPE base (e.g. 1000000.0 for Qwen2.5, 10000.0 default).

    Returns:
        The rotated x, same shape and dtype.
    """
    return torch.ops.pulsar.rope(x, pos, theta)


def reposition_kv(
    k_pool: torch.Tensor,
    slots: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    theta: float,
) -> None:
    """Reposition already-RoPE'd keys in a paged KV pool, HF/Qwen2 convention.

    The pool stores roped keys. Moving a key to a new pos is one extra
    rotation by the delta angle, so
    reposition(rope(raw, p_old), p_old -> p_new) == rope(raw, p_new). RoPE
    applies to K only; there is no V variant. The delta angle is formed in fp64
    and the rotation runs in fp32, matching rope.

    Args:
        k_pool: One layer's key pool, [num_pages, page_size, n_kv_heads,
            head_dim], contiguous. Mutated in place.
        slots: int32 [M]. Physical slot of each key, indexing the flattened
            [num_pages*page_size] first two dims (slot = page*page_size +
            offset).
        old_positions: int32 [M]. The pos each key is currently roped at.
        new_positions: int32 [M]. The pos to rotate each key to.
        theta: RoPE base (must match the base the keys were roped with).

    Returns:
        None. k_pool is updated in place.
    """
    torch.ops.pulsar.reposition_kv(k_pool, slots, old_positions, new_positions, theta)
