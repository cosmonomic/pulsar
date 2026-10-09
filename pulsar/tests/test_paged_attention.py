"""torch.ops.pulsar.write_kv scatters new K/V rows into a shared page pool.
torch.ops.pulsar.attn_decode does one-query-per-sequence decode attention over
ragged views gathered through per-sequence page tables. The pool holds unrotated
keys; the op rotates each to its position under the sequence's RoPE layout and
accumulates a per-key mass into the sequence's own view rows.

These tests compare against a plain-PyTorch reference that rotates the gathered
keys with ops.rope, over shuffled lines where two views share a line, under both
the contiguous and the compacted layout. For fp16/bf16, attn_decode's tensor-core
result is also checked against the scalar reference op per mass slot (not just
aggregate error), so a wrong fragment-to-slot mapping cannot hide behind an
aggregate check.
Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

DTYPES = [torch.float32, torch.bfloat16]

# Decode is validated for fp32 (scalar path) and fp16/bf16 (tensor-core path).
DECODE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# (n_q_heads, n_kv_heads, head_dim)
HEAD_CONFIGS = [
    (4, 4, 64),  # MHA
    (8, 2, 64),  # GQA
    (16, 16, 128),  # MHA, head_dim 128
    (12, 4, 128),  # GQA, head_dim 128
]

PAGE_SIZES = [16, 32]

# Forced split counts for the flash-decode path. 1 exercises the 3-kernel path at a
# single split; the larger counts (with the short CONTEXT_LENS below) produce splits
# that exceed some sequences' tile counts, covering the empty-split path.
SPLIT_COUNTS = [1, 2, 3, 4, 8]

# Ragged context lengths; includes non-multiples of page_size and > one page.
CONTEXT_LENS = [
    [1, 7, 16, 33, 40, 65, 100, 129],
    [3, 15, 17, 48, 63],
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


def _check_write(dtype, page_size):
    torch.manual_seed(0)
    num_pages = 20
    n_kv_heads = 4
    head_dim = 64
    num_new = 37

    k_pool = torch.randn(
        num_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_pool = torch.randn(
        num_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    k_pool_orig = k_pool.clone()
    v_pool_orig = v_pool.clone()

    total_slots = num_pages * page_size
    slots = torch.randperm(total_slots, device="cuda")[:num_new].to(torch.int32)

    k_new = torch.randn(num_new, n_kv_heads, head_dim, device="cuda", dtype=dtype)
    v_new = torch.randn(num_new, n_kv_heads, head_dim, device="cuda", dtype=dtype)

    ops.write_kv(k_pool, v_pool, k_new, v_new, slots)

    flat_k = k_pool.view(total_slots, n_kv_heads, head_dim)
    flat_v = v_pool.view(total_slots, n_kv_heads, head_dim)
    written_ok = torch.equal(flat_k[slots.long()], k_new) and torch.equal(
        flat_v[slots.long()], v_new
    )

    mask = torch.ones(total_slots, dtype=torch.bool, device="cuda")
    mask[slots.long()] = False
    flat_k_orig = k_pool_orig.view(total_slots, n_kv_heads, head_dim)
    flat_v_orig = v_pool_orig.view(total_slots, n_kv_heads, head_dim)
    untouched_ok = torch.equal(flat_k[mask], flat_k_orig[mask]) and torch.equal(
        flat_v[mask], flat_v_orig[mask]
    )

    return written_ok and untouched_ok


def _build_views(head_cfg, page_size, context_lens, dtype, seed):
    """Builds a pool of unrotated K/V and one view per sequence over shuffled lines.
    View 1 takes view 0's first line as its own first page, so the two share it at
    different view positions; odd sequences use the compacted layout."""
    torch.manual_seed(seed)
    n_q_heads, n_kv_heads, head_dim = head_cfg
    num_seqs = len(context_lens)

    per_seq_pages = [_pages_needed(c, page_size) for c in context_lens]
    max_pages = max(per_seq_pages)
    total_pages = sum(per_seq_pages) + 3  # a few spare lines

    perm = torch.randperm(total_pages).tolist()
    seq_lines = []
    cursor = 0
    for nb in per_seq_pages:
        seq_lines.append(perm[cursor : cursor + nb])
        cursor += nb
    if num_seqs > 1:
        seq_lines[1][0] = seq_lines[0][0]
    page_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32)
    for s, lines in enumerate(seq_lines):
        page_tables[s, : len(lines)] = torch.tensor(lines, dtype=torch.int32)

    layouts = [_layout(ctx, s % 2 == 1) for s, ctx in enumerate(context_lens)]
    cu_view_pages = [0]
    for nb in per_seq_pages:
        cu_view_pages.append(cu_view_pages[-1] + nb)

    k_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    q = torch.randn(num_seqs, n_q_heads, head_dim, device="cuda", dtype=dtype)
    return dict(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        page_tables=page_tables.to("cuda"),
        context_lens=torch.tensor(context_lens, dtype=torch.int32, device="cuda"),
        rope_layout=torch.tensor(layouts, dtype=torch.int32, device="cuda"),
        cu_view_pages=torch.tensor(cu_view_pages, dtype=torch.int32, device="cuda"),
        seq_lines=seq_lines,
        layouts=layouts,
        total_view_pages=cu_view_pages[-1],
    )


def _view_kv(views, s, ctx):
    """Sequence s's keys rotated to their layout positions by ops.rope (stored back
    in the pool dtype, as the kernels round them) and its values, both fp32
    [ctx, n_kv_heads, head_dim]."""
    k_pool, v_pool = views["k_pool"], views["v_pool"]
    n_kv_heads, head_dim = k_pool.shape[2], k_pool.shape[3]
    lines = torch.tensor(views["seq_lines"][s], device="cuda")
    k = k_pool[lines].reshape(-1, n_kv_heads, head_dim)[:ctx]
    v = v_pool[lines].reshape(-1, n_kv_heads, head_dim)[:ctx]
    pos = torch.tensor(
        _positions(views["layouts"][s], ctx), dtype=torch.int64, device="cuda"
    )
    k = ops.rope(k.transpose(0, 1).contiguous(), pos, THETA).transpose(0, 1)
    return k.float(), v.float()


def _reference(head_cfg, page_size, context_lens, views, scale):
    n_q_heads, n_kv_heads, head_dim = head_cfg
    group = n_q_heads // n_kv_heads
    q = views["q"]
    cu_view_pages = views["cu_view_pages"].tolist()

    o_ref = torch.empty(
        len(context_lens), n_q_heads, head_dim, device="cuda", dtype=torch.float32
    )
    mass_ref = torch.zeros(
        views["total_view_pages"], page_size, n_q_heads, device="cuda",
        dtype=torch.float32,
    )
    mass_rows = mass_ref.view(-1, n_q_heads)

    for s, ctx in enumerate(context_lens):
        ks, vs = _view_kv(views, s, ctx)
        first_row = cu_view_pages[s] * page_size
        for h in range(n_q_heads):
            g = h // group
            scores = scale * (ks[:, g, :] @ q[s, h].float())  # [ctx]
            w = torch.softmax(scores, dim=0)
            o_ref[s, h] = w @ vs[:, g, :]
            mass_rows[first_row : first_row + ctx, h] += w * ctx
    return o_ref, mass_ref


def _decode(views, scale, mass, decay, op=None, **kwargs):
    op = op or ops.attn_decode
    return op(
        views["q"], views["k_pool"], views["v_pool"], views["page_tables"],
        views["context_lens"], views["rope_layout"], THETA, scale, mass,
        views["cu_view_pages"], decay, **kwargs,
    )


def _check_decode(head_cfg, page_size, context_lens, dtype):
    views = _build_views(head_cfg, page_size, context_lens, dtype, seed=1)
    head_dim = head_cfg[2]
    scale = 1.0 / (head_dim**0.5)

    def new_mass():
        return torch.zeros(
            views["total_view_pages"], page_size, head_cfg[0], device="cuda",
            dtype=torch.float32,
        )

    # decay=1.0 disables the EMA gain, so mass accumulates the raw normalized
    # weight the reference computes.
    decay = torch.ones(len(context_lens), device="cuda", dtype=torch.float32)
    # ops.attn_decode runs the tensor-core kernel for fp16/bf16, scalar for fp32.
    mass = new_mass()
    o = _decode(views, scale, mass, decay)
    mass_scalar = new_mass()
    o_scalar = _decode(
        views, scale, mass_scalar, decay, op=torch.ops.pulsar.attn_decode_scalar
    )
    o_ref, mass_ref = _reference(head_cfg, page_size, context_lens, views, scale)

    tols = _tols(dtype)
    o_err = (o.float() - o_ref).abs().max().item()
    m_err = (mass - mass_ref).abs().max().item()
    o_ok = torch.allclose(o.float(), o_ref, **tols)
    m_ok = torch.allclose(mass, mass_ref, atol=1e-3, rtol=1e-3)

    # Mass compared per slot (both fp32) against the scalar oracle; no slot may differ
    # beyond fp32 accumulation noise.
    o_vs_scalar = (o.float() - o_scalar.float()).abs().max().item()
    m_vs_scalar = (mass - mass_scalar).abs().max().item()
    o_sc_ok = torch.allclose(o.float(), o_scalar.float(), **tols)
    m_sc_ok = torch.allclose(mass, mass_scalar, atol=1e-4, rtol=1e-4)

    # Forced-split (flash-decode) path at several split counts. For fp32 the op
    # routes to the scalar kernel, so it trivially matches.
    split_m_vs_scalar = 0.0
    for num_splits in SPLIT_COUNTS:
        mass_split = new_mass()
        o_split = _decode(views, scale, mass_split, decay, num_splits=num_splits)
        o_ok = o_ok and torch.allclose(o_split.float(), o_ref, **tols)
        m_ok = m_ok and torch.allclose(mass_split, mass_ref, atol=1e-3, rtol=1e-3)
        o_sc_ok = o_sc_ok and torch.allclose(o_split.float(), o_scalar.float(), **tols)
        m_sc_ok = m_sc_ok and torch.allclose(
            mass_split, mass_scalar, atol=1e-4, rtol=1e-4
        )
        split_m_vs_scalar = max(
            split_m_vs_scalar, (mass_split - mass_scalar).abs().max().item()
        )

    ok = o_ok and m_ok and o_sc_ok and m_sc_ok
    return ok, o_err, m_err, o_vs_scalar, m_vs_scalar, split_m_vs_scalar


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_write_kv(dtype, page_size):
    assert _check_write(dtype, page_size), f"dtype={dtype} page_size={page_size}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DECODE_DTYPES)
@pytest.mark.parametrize("page_size", PAGE_SIZES)
@pytest.mark.parametrize("head_cfg", HEAD_CONFIGS)
@pytest.mark.parametrize("context_lens", CONTEXT_LENS)
def test_paged_decode(dtype, page_size, head_cfg, context_lens):
    ok, o_err, m_err, o_sc, m_sc, split_m_sc = _check_decode(
        head_cfg, page_size, context_lens, dtype
    )
    assert ok, (
        f"head_cfg={head_cfg} page_size={page_size} dtype={dtype} "
        f"ctx={context_lens} o_maxerr={o_err:.2e} mass_maxerr={m_err:.2e} "
        f"o_vs_scalar={o_sc:.2e} mass_vs_scalar(per_slot)={m_sc:.2e} "
        f"split_mass_vs_scalar(per_slot)={split_m_sc:.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_skips_mass(dtype):
    """Without a mass the op computes the same o. The scalar kernel combines its
    threads' outputs in atomic order, so fp32 agrees to rounding, not to the bit."""
    head_cfg = (8, 2, 64)
    views = _build_views(head_cfg, 16, [40, 70], dtype, seed=2)
    scale = 1.0 / (head_cfg[2] ** 0.5)
    mass = torch.zeros(
        views["total_view_pages"], 16, head_cfg[0], device="cuda", dtype=torch.float32
    )
    decay = torch.ones(2, device="cuda", dtype=torch.float32)
    with_mass = _decode(views, scale, mass, decay)
    without = ops.attn_decode(
        views["q"], views["k_pool"], views["v_pool"], views["page_tables"],
        views["context_lens"], views["rope_layout"], THETA, scale,
    )
    assert torch.allclose(with_mass, without, atol=1e-6, rtol=1e-5)


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


def _opcheck_decode_inputs(dtype):
    page_size = 16
    q = torch.randn(3, 4, 8, device="cuda", dtype=dtype)
    k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    page_tables = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=torch.int32, device="cuda"
    )
    context_lens = torch.tensor([5, 20, 33], dtype=torch.int32, device="cuda")
    rope_layout = torch.tensor(
        [[0, 0, -1], [2, 10, 4], [0, 0, -1]], dtype=torch.int32, device="cuda"
    )
    mass = torch.zeros(9, page_size, 4, device="cuda", dtype=torch.float32)
    cu_view_pages = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device="cuda")
    decay = torch.ones(3, dtype=torch.float32, device="cuda")
    return (q, k_pool, v_pool, page_tables, context_lens, rope_layout, THETA, 0.35,
            mass, cu_view_pages, decay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_write():
    for dtype in DTYPES:
        page_size = 16
        k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        k_new = torch.randn(5, 2, 8, device="cuda", dtype=dtype)
        v_new = torch.randn(5, 2, 8, device="cuda", dtype=dtype)
        slots = torch.tensor([0, 17, 3, 40, 9], dtype=torch.int32, device="cuda")
        torch.library.opcheck(
            torch.ops.pulsar.write_kv,
            (k_pool, v_pool, k_new, v_new, slots),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_decode():
    for dtype in DTYPES:
        torch.library.opcheck(
            torch.ops.pulsar.attn_decode,
            _opcheck_decode_inputs(dtype),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_decode_split():
    for dtype in DTYPES:
        torch.library.opcheck(
            torch.ops.pulsar.attn_decode,
            _opcheck_decode_inputs(dtype),
            {"num_splits": 2},
            test_utils=_OPCHECK_UTILS,
        )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True

    for dtype in DTYPES:
        for page_size in PAGE_SIZES:
            ok = _check_write(dtype, page_size)
            all_pass &= ok
            print(
                f"[{'PASS' if ok else 'FAIL'}] write   {str(dtype):15s} "
                f"page_size={page_size}"
            )

    for dtype in DECODE_DTYPES:
        for page_size in PAGE_SIZES:
            for head_cfg in HEAD_CONFIGS:
                for ctx in CONTEXT_LENS:
                    ok, o_err, m_err, o_sc, m_sc, split_m_sc = _check_decode(
                        head_cfg, page_size, ctx, dtype
                    )
                    all_pass &= ok
                    print(
                        f"[{'PASS' if ok else 'FAIL'}] decode  {str(dtype):15s} "
                        f"bs={page_size} heads={head_cfg} "
                        f"o={o_err:.2e} m={m_err:.2e} "
                        f"o_vs_sc={o_sc:.2e} m_slot_vs_sc={m_sc:.2e} "
                        f"split_m_slot_vs_sc={split_m_sc:.2e}"
                    )

    for name, fn in (("write", test_opcheck_write), ("decode", test_opcheck_decode)):
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
