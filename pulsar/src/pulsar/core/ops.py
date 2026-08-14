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


def _attn_causal_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mass: torch.Tensor,
    scale: float,
    causal_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(attention_mass)


def _attn_causal_cache_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attention_mass: torch.Tensor,
    scale: float,
    cur_len: int,
) -> torch.Tensor:
    # attention_mass is mutated in place (Tensor(a!)); the op returns only o.
    return torch.empty_like(q)


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
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    lse_capture: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    # mass_pool/lse_capture are mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_decode_scalar_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    # mass_pool is mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_decode_split_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_splits: int,
    attention_mass_decay: torch.Tensor,
    lse_capture: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    # mass_pool/lse_capture are mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_prefill_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    lse_capture: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    # mass_pool/lse_capture are mutated in place; the op returns only o.
    return torch.empty_like(q)


def _attn_prefill_scalar_fake(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    # mass_pool is mutated in place; the op returns only o.
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
        "pulsar::attn_causal": _attn_causal_fake,
        "pulsar::attn_causal_cache": _attn_causal_cache_fake,
        "pulsar::write_kv": _write_kv_fake,
        "pulsar::attn_decode": _attn_decode_fake,
        # _scalar/_split are internal validation seams (scalar oracle, forced
        # split), registered here for tests; pulsar.ops has no public wrapper
        # for either.
        "pulsar::attn_decode_scalar": _attn_decode_scalar_fake,
        "pulsar::attn_decode_split": _attn_decode_split_fake,
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


def attn_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mass: torch.Tensor,
    scale: float,
    causal_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused causal attention with a per-key attention_mass side output.

    Args:
        q: Query tokens for this step, [n_q_heads, seq_q, head_dim].
        k: Key cache slice, [n_kv_heads, L, head_dim].
        v: Value cache slice, [n_kv_heads, L, head_dim].
        attention_mass: Running per-key signal, [n_kv_heads, L], fp32. Added to.
        scale: Softmax scale; softmax(scale * q @ k^T).
        causal_offset: Pos of the first query row. Query row i may
            attend to key j iff j <= causal_offset + i.

    Returns:
        (o, attention_mass_out) where o is [n_q_heads, seq_q, head_dim] and
        attention_mass_out is attention_mass plus the normalized attention mass each key
        received this step (summed over the query heads and rows sharing it).
    """
    return torch.ops.pulsar.attn_causal(q, k, v, attention_mass, scale, causal_offset)


def attn_causal_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    attention_mass: torch.Tensor,
    scale: float,
    cur_len: int,
) -> torch.Tensor:
    """Cache-form fused causal attention over fixed-capacity KV buffers.

    The new tokens' K/V are already written into k_cache/v_cache at rows
    [cur_len, cur_len + seq_q) before this call. Only rows [0, valid_len) with
    valid_len = cur_len + seq_q are read; capacity beyond valid_len may hold
    garbage and is ignored.

    Args:
        q: Query tokens for this step, [n_q_heads, seq_q, head_dim].
        k_cache: Fixed-capacity key buffer, [n_kv_heads, max_len, head_dim].
        v_cache: Fixed-capacity value buffer, [n_kv_heads, max_len, head_dim].
        attention_mass: Running per-key signal, [n_kv_heads, max_len], fp32.
            Mutated in place: this step's normalized attention mass is added into
            rows [0, valid_len); rows beyond valid_len are untouched.
        scale: Softmax scale; softmax(scale * q @ k^T).
        cur_len: Number of already-cached tokens before this step. Query row i
            (pos cur_len + i) attends to key j iff j <= cur_len + i.

    Returns:
        o, [n_q_heads, seq_q, head_dim]. attention_mass is updated in place and
        not returned.
    """
    return torch.ops.pulsar.attn_causal_cache(
        q, k_cache, v_cache, attention_mass, scale, cur_len
    )


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
        k_new: New keys, [num_new_tokens, n_kv_heads, head_dim].
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
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    lse_capture: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    """Paged decode attention with a per-key mass side output (seq_q == 1).

    Each sequence contributes one already-RoPE'd new-token query that attends
    over its whole context (non-causal). Keys are gathered through the sequence's
    page table.

    Args:
        q: Query tokens, [num_seqs, n_q_heads, head_dim].
        k_pool: Key pool, [num_pages, page_size, n_kv_heads, head_dim].
        v_pool: Value pool, same shape as k_pool.
        mass_pool: Running per-key signal, [num_pages, page_size, n_q_heads],
            fp32. Mutated in place: this step's normalized attention mass is added
            into the slots each sequence attends, each query head into its own
            column (per query head, no group sum).
        page_tables: int32 [num_seqs, max_pages]. page_tables[s][b] is the
            physical page id of logical page b of sequence s.
        context_lens: int32 [num_seqs]. Number of valid keys for each sequence.
        scale: Softmax scale; softmax(scale * q @ k^T).
        attention_mass_decay: fp32 [num_seqs] on the pool device. Sequence s's EMA
            gain alpha, the factor its per-key mass term is scaled by; 1.0
            accumulates the raw normalized weight. One entry per sequence, so
            sequences batched together may decay at different rates. Each key's mass
            is scaled by the length it is stated against, so the stored quantity is a
            multiple of a uniform share over that length.
        lse_capture: fp32 contiguous [num_seqs, n_q_heads], written in place with
            each row's logsumexp of scale * q @ k^T over the keys it attended (the
            softmax denominator the attention used). None leaves it unwritten.
        mass_length_gain: TOKENS. The length each key's mass is stated against, so
            uniform attention over exactly that many keys stores 1. 0 or less takes
            each sequence's own context length, which only the op can read. For any
            other CONSTANT length, pass 1 and multiply the result by it once
            afterwards, since the mass sums over query tokens and a constant
            factor commutes with that sum.

    Returns:
        o, [num_seqs, n_q_heads, head_dim]. mass_pool and lse_capture are updated in
        place and not returned.
    """
    return torch.ops.pulsar.attn_decode(
        q, k_pool, v_pool, mass_pool, page_tables, context_lens, scale,
        attention_mass_decay, lse_capture, mass_length_gain,
    )


def attn_prefill(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    mass_pool: torch.Tensor,
    page_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    scale: float,
    attention_mass_decay: torch.Tensor,
    lse_capture: torch.Tensor | None = None,
    mass_length_gain: float = 0.0,
) -> torch.Tensor:
    """Paged prefill attention with a per-key mass side output (causal, varlen).

    The prefill counterpart of attn_decode, with many query tokens per
    sequence, causal-masked, over a ragged batch. Sequence i's query
    tokens are q[cu_seqlens_q[i] : cu_seqlens_q[i+1]] and occupy context
    pos [ctx_start_i, seqlens_k[i]) with ctx_start_i = seqlens_k[i] -
    seq_q_i. A query token at context pos p attends to keys [0, p]
    (inclusive), gathered through the sequence's page table.

    Args:
        q: Query tokens, [total_q, n_q_heads, head_dim] (already RoPE'd).
        k_pool: Key pool, [num_pages, page_size, n_kv_heads, head_dim].
        v_pool: Value pool, same shape as k_pool.
        mass_pool: Running per-key signal, [num_pages, page_size, n_q_heads],
            fp32. Mutated in place: for each key at context pos kp and query head h,
            the sum over that head's query tokens with p >= kp of the normalized
            weight is added into its column (per query head, no group sum).
        page_tables: int32 [num_seqs, max_pages]. page_tables[i][b] is the
            physical page id of logical page b of sequence i.
        cu_seqlens_q: int32 [num_seqs+1]. Prefix sums of per-seq query lengths.
        seqlens_k: int32 [num_seqs]. Total context length per sequence (>= that
            sequence's query length).
        scale: Softmax scale; softmax(scale * q @ k^T).
        attention_mass_decay: fp32 [num_seqs] on the pool device. Sequence i's EMA
            gain alpha, the factor its per-key mass term is scaled by; 1.0
            accumulates the raw normalized weight. The within-chunk retention comes
            from the same value: a query token at offset o from the chunk end
            contributes (1 - alpha)^o. One entry per sequence, so sequences batched
            together may decay at different rates. Each query's contribution is
            scaled by the length it is stated against, so the stored quantity is a
            multiple of a uniform share over that length.
        lse_capture: fp32 contiguous [total_q, n_q_heads], written in place with each
            query row's logsumexp of scale * q @ k^T over its causal prefix, so the
            key count varies by row. None leaves it unwritten.
        mass_length_gain: TOKENS. The length each query's contribution is stated
            against, so uniform attention over exactly that many keys stores 1. 0 or
            less takes each query's own causal key count, which varies query by query
            and only the op can read. For any other CONSTANT length, pass 1 and
            multiply by it once afterwards.

    Returns:
        o, [total_q, n_q_heads, head_dim]. mass_pool and lse_capture are updated in
        place and not returned.
    """
    return torch.ops.pulsar.attn_prefill(
        q, k_pool, v_pool, mass_pool, page_tables, cu_seqlens_q, seqlens_k, scale,
        attention_mass_decay, lse_capture, mass_length_gain,
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
