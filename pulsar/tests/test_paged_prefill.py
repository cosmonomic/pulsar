"""torch.ops.pulsar.attn_prefill does causal, varlen prefill attention
(seq_q >= 1 query tokens per sequence) over per-sequence views of a paged KV
pool. The pool holds unrotated keys; the op rotates each to its position under
the sequence's RoPE layout and accumulates a per-key mass into the sequence's
own view rows.

These tests compare against a plain-PyTorch reference that rotates the gathered
keys with ops.rope, over a ragged batch mixing first prefill (ctx_start == 0)
and chunked prefill (ctx_start > 0, seqlens_k > seq_q), with query spans
crossing multiple 16-row tiles and KV pages, two views sharing a lane, and both
the contiguous and the compacted layout. For fp16/bf16, attn_prefill's
tensor-core result is also checked against the scalar reference op per mass
slot (not just aggregate error), so a wrong causal boundary or key mapping
cannot hide behind an aggregate check.
Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

# Prefill is validated for fp32 (scalar path) and fp16/bf16 (tensor-core path).
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# (n_q_heads, n_kv_heads, head_dim)
HEAD_CONFIGS = [
    (4, 4, 64),  # MHA
    (8, 2, 64),  # GQA
    (16, 16, 128),  # MHA, head_dim 128
    (12, 4, 128),  # GQA, head_dim 128
]

PAGE_SIZES = [16, 32]

# The EMA gain alpha the batched cases run at. It both scales each contribution and
# sets the within-chunk retention 1 - alpha, so every query token of a chunk carries a
# distinct weight and neither factor can hide the other.
MASS_DECAY = 0.25

# Varlen batches: list of (seq_q, seqlens_k) per sequence. Mixes first prefill
# (seqlens_k == seq_q, ctx_start == 0) and chunked prefill (seqlens_k > seq_q,
# ctx_start > 0), non-multiples of page_size, single-token queries, and query
# spans crossing multiple 16-row tiles and multiple KV pages.
BATCHES = [
    [(16, 16), (7, 40), (33, 33), (1, 65)],
    [(40, 40), (3, 20), (65, 100), (9, 9)],
    [(1, 1), (17, 17), (31, 48)],
]


THETA = 1000000.0


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    if dtype == torch.float16:
        return dict(atol=5e-3, rtol=5e-3)
    return dict(atol=1e-4, rtol=1e-4)


def _pages_needed(ctx, page_size):
    return (ctx + page_size - 1) // page_size


def _layout(ctx, compacted):
    """{n_sink, working_lo, short_offset}: the identity, or a compacted layout whose
    distant region is the middle of the view."""
    if not compacted:
        return (0, 0, -1)
    n_sink = min(4, ctx)
    working_lo = max(n_sink, ctx - 24)
    return (n_sink, working_lo, n_sink + 3)


def _positions(layout, ctx):
    n_sink, working_lo, short_offset = layout
    return [
        j if j < n_sink else short_offset if j < working_lo
        else short_offset + 1 + j - working_lo
        for j in range(ctx)
    ]


def _build_prefill(head_cfg, page_size, seq_specs, dtype, seed):
    """Builds a pool of unrotated K/V and one view per sequence over shuffled lanes,
    and concatenates the queries. View 1 takes view 0's first lane as its own first
    page, so the two share it at different view positions; odd sequences use the
    compacted layout."""
    torch.manual_seed(seed)
    n_q_heads, n_kv_heads, head_dim = head_cfg
    num_seqs = len(seq_specs)

    ctx_lens = [c for (_, c) in seq_specs]
    seq_qs = [sq for (sq, _) in seq_specs]
    per_seq_pages = [_pages_needed(c, page_size) for c in ctx_lens]
    max_pages = max(per_seq_pages)
    total_pages = sum(per_seq_pages) + 3  # a few spare lanes

    perm = torch.randperm(total_pages).tolist()
    seq_lanes = []
    cursor = 0
    for nb in per_seq_pages:
        seq_lanes.append(perm[cursor : cursor + nb])
        cursor += nb
    if num_seqs > 1:
        seq_lanes[1][0] = seq_lanes[0][0]
    page_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32)
    for s, lanes in enumerate(seq_lanes):
        page_tables[s, : len(lanes)] = torch.tensor(lanes, dtype=torch.int32)

    layouts = [_layout(ctx, s % 2 == 1) for s, ctx in enumerate(ctx_lens)]
    cu_view_pages = [0]
    for nb in per_seq_pages:
        cu_view_pages.append(cu_view_pages[-1] + nb)
    cu = [0]
    for sq in seq_qs:
        cu.append(cu[-1] + sq)

    k_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    q = torch.randn(cu[-1], n_q_heads, head_dim, device="cuda", dtype=dtype)
    return dict(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        page_tables=page_tables.to("cuda"),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device="cuda"),
        seqlens_k=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
        rope_layout=torch.tensor(layouts, dtype=torch.int32, device="cuda"),
        cu_view_pages=torch.tensor(cu_view_pages, dtype=torch.int32, device="cuda"),
        seq_lanes=seq_lanes,
        layouts=layouts,
        total_view_pages=cu_view_pages[-1],
    )


def _view_kv(views, s, ctx):
    """Sequence s's keys rotated to their layout positions by ops.rope (stored back
    in the pool dtype, as the kernels round them) and its values, both fp32
    [ctx, n_kv_heads, head_dim]."""
    k_pool, v_pool = views["k_pool"], views["v_pool"]
    n_kv_heads, head_dim = k_pool.shape[2], k_pool.shape[3]
    lanes = torch.tensor(views["seq_lanes"][s], device="cuda")
    k = k_pool[lanes].reshape(-1, n_kv_heads, head_dim)[:ctx]
    v = v_pool[lanes].reshape(-1, n_kv_heads, head_dim)[:ctx]
    pos = torch.tensor(
        _positions(views["layouts"][s], ctx), dtype=torch.int64, device="cuda"
    )
    k = ops.rope(k.transpose(0, 1).contiguous(), pos, THETA).transpose(0, 1)
    return k.float(), v.float()


def _reference(head_cfg, page_size, seq_specs, views, scale, alpha=1.0,
               retention=None):
    """Reference mass/o. retention defaults to the 1 - alpha the op derives; pass it
    explicitly only to model a grading the op cannot produce."""
    if retention is None:
        retention = 1.0 - alpha
    n_q_heads, n_kv_heads, head_dim = head_cfg
    group = n_q_heads // n_kv_heads
    q = views["q"]
    cu = views["cu_seqlens_q"].tolist()
    cu_view_pages = views["cu_view_pages"].tolist()

    o_ref = torch.empty(
        q.size(0), n_q_heads, head_dim, device="cuda", dtype=torch.float32
    )
    mass_ref = torch.zeros(
        views["total_view_pages"], page_size, n_q_heads, device="cuda",
        dtype=torch.float32,
    )
    mass_rows = mass_ref.view(-1, n_q_heads)

    for s, (seq_q, ctx) in enumerate(seq_specs):
        ks, vs = _view_kv(views, s, ctx)
        first_row = cu_view_pages[s] * page_size
        ctx_start = ctx - seq_q
        for j in range(seq_q):
            causal_key_count = ctx_start + j + 1
            global_q = cu[s] + j
            # wq is the gain alpha times retention graded by offset from the chunk
            # end (retention**0 for the last query); retention == 1 gives no grading.
            wq = alpha * retention ** (seq_q - 1 - j)
            for h in range(n_q_heads):
                g = h // group
                kg = ks[:causal_key_count, g, :]
                scores = scale * (kg @ q[global_q, h].float())
                w = torch.softmax(scores, dim=0)
                o_ref[global_q, h] = w @ vs[:causal_key_count, g, :]
                mass_rows[first_row : first_row + causal_key_count, h] += (
                    w * wq * causal_key_count
                )
    return o_ref, mass_ref


def _prefill(views, scale, mass, decay, op=None, **kwargs):
    op = op or ops.attn_prefill
    return op(
        views["q"], views["k_pool"], views["v_pool"], views["page_tables"],
        views["cu_seqlens_q"], views["seqlens_k"], views["rope_layout"], THETA, scale,
        mass, views["cu_view_pages"], decay, **kwargs,
    )


def _new_mass(views, page_size, n_q_heads):
    return torch.zeros(
        views["total_view_pages"], page_size, n_q_heads, device="cuda",
        dtype=torch.float32,
    )


def _check_prefill(head_cfg, page_size, seq_specs, dtype):
    views = _build_prefill(head_cfg, page_size, seq_specs, dtype, seed=1)
    head_dim = head_cfg[2]
    scale = 1.0 / (head_dim**0.5)

    # One EMA gain per sequence, all at MASS_DECAY here (the per-sequence spread is
    # covered by test_prefill_mass_decay).
    decay = torch.full(
        (len(seq_specs),), MASS_DECAY, device="cuda", dtype=torch.float32
    )
    # ops.attn_prefill runs the tensor-core kernel for fp16/bf16, scalar for fp32.
    mass = _new_mass(views, page_size, head_cfg[0])
    o = _prefill(views, scale, mass, decay)
    mass_scalar = _new_mass(views, page_size, head_cfg[0])
    o_scalar = _prefill(
        views, scale, mass_scalar, decay, op=torch.ops.pulsar.attn_prefill_scalar
    )
    o_ref, mass_ref = _reference(
        head_cfg, page_size, seq_specs, views, scale, alpha=MASS_DECAY
    )

    tols = _tols(dtype)
    o_err = (o.float() - o_ref).abs().max().item()
    m_err = (mass - mass_ref).abs().max().item()
    o_ok = torch.allclose(o.float(), o_ref, **tols)
    m_ok = torch.allclose(mass, mass_ref, atol=1e-3, rtol=1e-3)

    # Mass compared per slot (both fp32) against the scalar oracle.
    o_vs_scalar = (o.float() - o_scalar.float()).abs().max().item()
    m_vs_scalar = (mass - mass_scalar).abs().max().item()
    o_sc_ok = torch.allclose(o.float(), o_scalar.float(), **tols)
    m_sc_ok = torch.allclose(mass, mass_scalar, atol=1e-4, rtol=1e-4)

    ok = o_ok and m_ok and o_sc_ok and m_sc_ok
    return ok, o_err, m_err, o_vs_scalar, m_vs_scalar


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("head_cfg", HEAD_CONFIGS)
@pytest.mark.parametrize("batch", BATCHES)
def test_paged_prefill(dtype, page_size, head_cfg, batch):
    ok, o_err, m_err, o_sc, m_sc = _check_prefill(head_cfg, page_size, batch, dtype)
    assert ok, (
        f"head_cfg={head_cfg} page_size={page_size} dtype={dtype} "
        f"batch={batch} o_maxerr={o_err:.2e} mass_maxerr={m_err:.2e} "
        f"o_vs_scalar={o_sc:.2e} mass_vs_scalar(per_slot)={m_sc:.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_prefill_mass_decay(dtype):
    """The per-key mass is the within-chunk graded sum of attention scaled by
    the sequence's own EMA gain. A query at chunk-offset j contributes
    alpha * (1-alpha)**(seq_q-1-j), so older-in-chunk queries weigh less. alpha
    is per sequence, so two sequences in one batched call accumulate at their
    own rate, which a batch-wide scalar cannot express. Checked against a
    token-by-token reference for both the tensor-core (bf16) and scalar (fp32)
    mass passes."""
    head_cfg = (4, 2, 64)  # head_dim 64 -> TC-eligible for bf16
    page_size = 16
    alphas = [0.1, 0.5]
    seq_specs = [(12, 12), (12, 12)]  # two 12-query causal prefill chunks
    views = _build_prefill(head_cfg, page_size, seq_specs, dtype, seed=3)
    scale = 1.0 / (head_cfg[2] ** 0.5)

    decay = torch.tensor(alphas, device="cuda", dtype=torch.float32)
    mass = _new_mass(views, page_size, head_cfg[0])
    _prefill(views, scale, mass, decay)
    mass_sc = _new_mass(views, page_size, head_cfg[0])
    _prefill(views, scale, mass_sc, decay, op=torch.ops.pulsar.attn_prefill_scalar)

    tol = 3e-3 if dtype == torch.bfloat16 else 1e-4
    cu_view_pages = views["cu_view_pages"].tolist()
    # Each sequence is checked against a reference at its own alpha, over its own
    # view rows; one shared alpha would fail one of the two.
    for s, alpha in enumerate(alphas):
        _, ref = _reference(head_cfg, page_size, seq_specs, views, scale, alpha=alpha)
        # Mass with the same gain but no within-chunk grading (retention dropped).
        _, flat = _reference(head_cfg, page_size, seq_specs, views, scale,
                             alpha=alpha, retention=1.0)
        rows = slice(cu_view_pages[s], cu_view_pages[s + 1])
        assert torch.allclose(mass[rows], ref[rows], atol=tol, rtol=tol), (
            f"seq {s} graded mass "
            f"err={(mass[rows] - ref[rows]).abs().max().item():.2e}")
        assert torch.allclose(mass_sc[rows], ref[rows], atol=tol, rtol=tol), (
            f"seq {s} scalar graded mass "
            f"err={(mass_sc[rows] - ref[rows]).abs().max().item():.2e}")
        # The grading must actually change the mass, or the test is inert.
        assert (flat[rows] - ref[rows]).abs().max().item() > 1e-2, (
            f"seq {s}: the within-chunk retention had no effect")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_skips_mass(dtype):
    """Without a mass the op computes the same o. The scalar kernel combines its
    threads' outputs in atomic order, so fp32 agrees to rounding, not to the bit."""
    head_cfg = (8, 2, 64)
    views = _build_prefill(head_cfg, 16, BATCHES[0], dtype, seed=2)
    scale = 1.0 / (head_cfg[2] ** 0.5)
    decay = torch.ones(len(BATCHES[0]), device="cuda", dtype=torch.float32)
    with_mass = _prefill(views, scale, _new_mass(views, 16, head_cfg[0]), decay)
    without = ops.attn_prefill(
        views["q"], views["k_pool"], views["v_pool"], views["page_tables"],
        views["cu_seqlens_q"], views["seqlens_k"], views["rope_layout"], THETA, scale,
    )
    assert torch.allclose(with_mass, without, atol=1e-6, rtol=1e-5)


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


def _opcheck_inputs(dtype):
    page_size = 16
    # 2 seqs: first-prefill (seq_q=5, ctx=5) and chunked (seq_q=3, ctx=20).
    q = torch.randn(8, 4, 8, device="cuda", dtype=dtype)
    k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor([5, 20], dtype=torch.int32, device="cuda")
    rope_layout = torch.tensor([[0, 0, -1], [2, 10, 4]], dtype=torch.int32, device="cuda")
    mass = torch.zeros(6, page_size, 4, device="cuda", dtype=torch.float32)
    cu_view_pages = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    decay = torch.full((2,), 0.25, dtype=torch.float32, device="cuda")
    return (q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, rope_layout, THETA,
            0.35, mass, cu_view_pages, decay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_prefill():
    for dtype in (torch.float32, torch.bfloat16):
        torch.library.opcheck(
            torch.ops.pulsar.attn_prefill,
            _opcheck_inputs(dtype),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_prefill_scalar():
    for dtype in (torch.float32, torch.bfloat16):
        torch.library.opcheck(
            torch.ops.pulsar.attn_prefill_scalar,
            _opcheck_inputs(dtype),
            test_utils=_OPCHECK_UTILS,
        )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True
    for dtype in DTYPES:
        for page_size in PAGE_SIZES:
            for head_cfg in HEAD_CONFIGS:
                for batch in BATCHES:
                    ok, o_err, m_err, o_sc, m_sc = _check_prefill(
                        head_cfg, page_size, batch, dtype
                    )
                    all_pass &= ok
                    print(
                        f"[{'PASS' if ok else 'FAIL'}] prefill {str(dtype):15s} "
                        f"bs={page_size} heads={head_cfg} "
                        f"o={o_err:.2e} m={m_err:.2e} "
                        f"o_vs_sc={o_sc:.2e} m_slot_vs_sc={m_sc:.2e}"
                    )
    for name, fn in (
        ("prefill", test_opcheck_prefill),
        ("prefill_scalar", test_opcheck_prefill_scalar),
    ):
        try:
            fn()
            print(f"[PASS] opcheck {name} (schema/faketensor/aot_dispatch_dynamic)")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] opcheck {name}: {exc}")
            all_pass = False
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
