"""torch.ops.pulsar.attn_prefill does causal, varlen prefill attention
(seq_q >= 1 query tokens per sequence) over a paged KV pool gathered through
per-sequence page tables, accumulating a per-key mass into the pool in place.

These tests compare against a plain-PyTorch paged reference, over a ragged
batch mixing first prefill (ctx_start == 0) and chunked prefill (ctx_start > 0,
seqlens_k > seq_q), with query spans crossing multiple 16-row tiles and KV
pages. For fp16/bf16, attn_prefill's tensor-core result is also checked
against the scalar reference op per physical slot of mass_pool (not just
aggregate error), so a wrong causal boundary or key mapping cannot hide behind
an aggregate check.
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


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    if dtype == torch.float16:
        return dict(atol=5e-3, rtol=5e-3)
    return dict(atol=1e-4, rtol=1e-4)


def _pages_needed(ctx, page_size):
    return (ctx + page_size - 1) // page_size


def _build_prefill(head_cfg, page_size, seq_specs, dtype, seed):
    """Builds a pool and page tables with disjoint shuffled physical pages per
    sequence, writes random K/V for all seqlens_k keys, concatenates the
    queries, and returns per-sequence gathered logical K/V for the reference."""
    torch.manual_seed(seed)
    n_q_heads, n_kv_heads, head_dim = head_cfg
    num_seqs = len(seq_specs)

    ctx_lens = [c for (_, c) in seq_specs]
    seq_qs = [sq for (sq, _) in seq_specs]
    per_seq_pages = [_pages_needed(c, page_size) for c in ctx_lens]
    max_pages = max(per_seq_pages)
    total_pages = sum(per_seq_pages) + 3  # a few spare pages

    perm = torch.randperm(total_pages).tolist()
    page_tables = torch.zeros(num_seqs, max_pages, dtype=torch.int32)
    cursor = 0
    seq_phys = []
    for s, nb in enumerate(per_seq_pages):
        phys = perm[cursor : cursor + nb]
        cursor += nb
        seq_phys.append(phys)
        for b in range(nb):
            page_tables[s, b] = phys[b]

    k_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_pool = torch.randn(
        total_pages, page_size, n_kv_heads, head_dim, device="cuda", dtype=dtype
    )

    ref_k, ref_v = [], []
    for s, ctx in enumerate(ctx_lens):
        ks = torch.empty(ctx, n_kv_heads, head_dim, device="cuda", dtype=dtype)
        vs = torch.empty(ctx, n_kv_heads, head_dim, device="cuda", dtype=dtype)
        for j in range(ctx):
            phys = seq_phys[s][j // page_size]
            off = j % page_size
            ks[j] = k_pool[phys, off]
            vs[j] = v_pool[phys, off]
        ref_k.append(ks)
        ref_v.append(vs)

    total_q = sum(seq_qs)
    q = torch.randn(total_q, n_q_heads, head_dim, device="cuda", dtype=dtype)
    cu = [0]
    for sq in seq_qs:
        cu.append(cu[-1] + sq)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor(ctx_lens, dtype=torch.int32, device="cuda")
    page_tables = page_tables.to("cuda")
    return (
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        seq_phys,
        ref_k,
        ref_v,
    )


def _reference(
    head_cfg, page_size, seq_specs, q, cu, seq_phys, ref_k, ref_v, total_pages,
    scale, alpha=1.0, retention=None
):
    """Reference mass/o. retention defaults to the 1 - alpha the op derives; pass it
    explicitly only to model a grading the op cannot produce."""
    if retention is None:
        retention = 1.0 - alpha
    n_q_heads, n_kv_heads, head_dim = head_cfg
    group = n_q_heads // n_kv_heads
    total_q = q.size(0)

    o_ref = torch.empty(
        total_q, n_q_heads, head_dim, device="cuda", dtype=torch.float32
    )
    mass_ref = torch.zeros(
        total_pages, page_size, n_q_heads, device="cuda", dtype=torch.float32
    )

    for s, (seq_q, ctx) in enumerate(seq_specs):
        ctx_start = ctx - seq_q
        q0 = cu[s]
        ks = ref_k[s].float()  # [ctx, n_kv_heads, head_dim]
        vs = ref_v[s].float()
        for j in range(seq_q):
            causal_key_count = ctx_start + j + 1
            global_q = q0 + j
            for h in range(n_q_heads):
                g = h // group
                kg = ks[:causal_key_count, g, :]  # [causal_key_count, head_dim]
                vg = vs[:causal_key_count, g, :]
                scores = scale * (q[global_q, h].float() @ kg.t())  # [causal_key_count]
                w = torch.softmax(scores, dim=0)
                o_ref[global_q, h] = w @ vg
                # wq is the gain alpha times retention graded by offset from the
                # chunk end (retention**0 for the last query); retention == 1 gives
                # no grading.
                wq = alpha * retention ** (seq_q - 1 - j)
                for kp in range(causal_key_count):
                    phys = seq_phys[s][kp // page_size]
                    off = kp % page_size
                    mass_ref[phys, off, h] += w[kp] * wq * causal_key_count  # own column for h
    return o_ref, mass_ref


def _check_prefill(head_cfg, page_size, seq_specs, dtype):
    (
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        seq_phys,
        ref_k,
        ref_v,
    ) = _build_prefill(head_cfg, page_size, seq_specs, dtype, seed=1)
    total_pages = k_pool.size(0)
    head_dim = head_cfg[2]
    scale = 1.0 / (head_dim**0.5)

    # ops.attn_prefill runs the tensor-core kernel for fp16/bf16, scalar for fp32.
    # Mass is per query head (head_cfg[0]); each head writes its own column.
    mass_pool = torch.zeros(
        total_pages, page_size, head_cfg[0], device="cuda", dtype=torch.float32
    )
    # One EMA gain per sequence, all at MASS_DECAY here (the per-sequence spread is
    # covered by test_prefill_mass_decay).
    decay = torch.full(
        (len(seq_specs),), MASS_DECAY, device="cuda", dtype=torch.float32
    )
    o = ops.attn_prefill(
        q, k_pool, v_pool, mass_pool, page_tables, cu_seqlens_q, seqlens_k, scale,
        decay,
    )

    # Scalar oracle op on the same inputs (fresh mass pool).
    mass_scalar = torch.zeros_like(mass_pool)
    o_scalar = torch.ops.pulsar.attn_prefill_scalar(
        q, k_pool, v_pool, mass_scalar, page_tables, cu_seqlens_q, seqlens_k, scale,
        decay,
    )

    cu = cu_seqlens_q.tolist()
    o_ref, mass_ref = _reference(
        head_cfg,
        page_size,
        seq_specs,
        q,
        cu,
        seq_phys,
        ref_k,
        ref_v,
        total_pages,
        scale,
        alpha=MASS_DECAY,
    )

    tols = _tols(dtype)
    o_err = (o.float() - o_ref).abs().max().item()
    m_err = (mass_pool - mass_ref).abs().max().item()
    o_ok = torch.allclose(o.float(), o_ref, atol=tols["atol"], rtol=tols["rtol"])
    m_ok = torch.allclose(mass_pool, mass_ref, atol=1e-3, rtol=1e-3)

    # Mass compared per physical slot (both fp32) against the scalar oracle.
    o_vs_scalar = (o.float() - o_scalar.float()).abs().max().item()
    m_vs_scalar = (mass_pool - mass_scalar).abs().max().item()
    o_sc_ok = torch.allclose(
        o.float(), o_scalar.float(), atol=tols["atol"], rtol=tols["rtol"]
    )
    m_sc_ok = torch.allclose(mass_pool, mass_scalar, atol=1e-4, rtol=1e-4)

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
    (q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, seq_phys, ref_k,
     ref_v) = _build_prefill(head_cfg, page_size, seq_specs, dtype, seed=3)
    total_pages = k_pool.size(0)
    scale = 1.0 / (head_cfg[2] ** 0.5)

    mass = torch.zeros(total_pages, page_size, head_cfg[0], device="cuda",
                       dtype=torch.float32)
    decay = torch.tensor(alphas, device="cuda", dtype=torch.float32)
    ops.attn_prefill(q, k_pool, v_pool, mass, page_tables, cu_seqlens_q, seqlens_k,
                     scale, decay)
    mass_sc = torch.zeros_like(mass)
    torch.ops.pulsar.attn_prefill_scalar(q, k_pool, v_pool, mass_sc, page_tables,
                                       cu_seqlens_q, seqlens_k, scale, decay)

    cu = cu_seqlens_q.tolist()
    tol = 3e-3 if dtype == torch.bfloat16 else 1e-4
    # Each sequence is checked against a reference at its own alpha, over its own
    # physical pages; one shared alpha would fail one of the two.
    for s, alpha in enumerate(alphas):
        _, ref = _reference(head_cfg, page_size, seq_specs, q, cu, seq_phys, ref_k,
                            ref_v, total_pages, scale, alpha=alpha)
        # Mass with the same gain but no within-chunk grading (retention dropped).
        _, flat = _reference(head_cfg, page_size, seq_specs, q, cu, seq_phys, ref_k,
                             ref_v, total_pages, scale, alpha=alpha, retention=1.0)
        pages = seq_phys[s]
        assert torch.allclose(mass[pages], ref[pages], atol=tol, rtol=tol), (
            f"seq {s} graded mass "
            f"err={(mass[pages] - ref[pages]).abs().max().item():.2e}")
        assert torch.allclose(mass_sc[pages], ref[pages], atol=tol, rtol=tol), (
            f"seq {s} scalar graded mass "
            f"err={(mass_sc[pages] - ref[pages]).abs().max().item():.2e}")
        # The grading must actually change the mass, or the test is inert.
        assert (flat[pages] - ref[pages]).abs().max().item() > 1e-2, (
            f"seq {s}: the within-chunk retention had no effect")


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


def _opcheck_inputs(dtype):
    page_size = 16
    # 2 seqs: first-prefill (seq_q=5, ctx=5) and chunked (seq_q=3, ctx=20).
    q = torch.randn(8, 4, 8, device="cuda", dtype=dtype)
    k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
    mass_pool = torch.zeros(8, page_size, 4, device="cuda", dtype=torch.float32)
    page_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 5, 8], dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor([5, 20], dtype=torch.int32, device="cuda")
    decay = torch.full((2,), 0.25, dtype=torch.float32, device="cuda")
    return (q, k_pool, v_pool, mass_pool, page_tables, cu_seqlens_q, seqlens_k, 0.35,
            decay)


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
