"""torch.ops.pulsar.write_kv scatters new K/V rows into a shared page pool.
torch.ops.pulsar.attn_decode does one-query-per-sequence decode attention over
ragged contexts gathered through per-sequence page tables, accumulating a
per-key mass into the pool in place.

These tests compare against a plain-PyTorch paged reference, with shuffled
non-contiguous physical pages to exercise the page-table indirection. For
fp16/bf16, attn_decode's tensor-core result is also checked against the
scalar reference op per physical slot of mass_pool (not just aggregate error),
so a wrong fragment-to-slot mapping cannot hide behind an aggregate check.
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

# Forced split counts for the flash-decode split op. 1 exercises the 3-kernel
# path at a single split; the larger counts (with the short CONTEXT_LENS below)
# produce splits that exceed some sequences' tile counts, covering the
# empty-split path.
SPLIT_COUNTS = [1, 2, 3, 4, 8]

# Ragged context lengths; includes non-multiples of page_size and > one page.
CONTEXT_LENS = [
    [1, 7, 16, 33, 40, 65, 100, 129],
    [3, 15, 17, 48, 63],
]


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    if dtype == torch.float16:
        return dict(atol=5e-3, rtol=5e-3)
    return dict(atol=1e-4, rtol=1e-4)


def _pages_needed(ctx, page_size):
    return (ctx + page_size - 1) // page_size


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


def _build_paged(head_cfg, page_size, context_lens, dtype, seed):
    """Builds a pool and page tables with disjoint shuffled physical pages per
    sequence, writes random K/V, and returns per-sequence contiguous logical K/V
    for the reference."""
    torch.manual_seed(seed)
    n_q_heads, n_kv_heads, head_dim = head_cfg
    num_seqs = len(context_lens)

    per_seq_pages = [_pages_needed(c, page_size) for c in context_lens]
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
    for s, ctx in enumerate(context_lens):
        ks = torch.empty(ctx, n_kv_heads, head_dim, device="cuda", dtype=dtype)
        vs = torch.empty(ctx, n_kv_heads, head_dim, device="cuda", dtype=dtype)
        for j in range(ctx):
            phys = seq_phys[s][j // page_size]
            off = j % page_size
            ks[j] = k_pool[phys, off]
            vs[j] = v_pool[phys, off]
        ref_k.append(ks)
        ref_v.append(vs)

    q = torch.randn(num_seqs, n_q_heads, head_dim, device="cuda", dtype=dtype)
    context_lens_t = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    page_tables = page_tables.to("cuda")
    return (q, k_pool, v_pool, page_tables, context_lens_t, seq_phys, ref_k, ref_v)


def _reference(
    head_cfg, page_size, context_lens, q, seq_phys, ref_k, ref_v, total_pages, scale
):
    n_q_heads, n_kv_heads, head_dim = head_cfg
    group = n_q_heads // n_kv_heads
    num_seqs = len(context_lens)

    o_ref = torch.empty(
        num_seqs, n_q_heads, head_dim, device="cuda", dtype=torch.float32
    )
    mass_ref = torch.zeros(
        total_pages, page_size, n_q_heads, device="cuda", dtype=torch.float32
    )

    for s, ctx in enumerate(context_lens):
        ks = ref_k[s].float()  # [ctx, n_kv_heads, head_dim]
        vs = ref_v[s].float()
        for h in range(n_q_heads):
            g = h // group
            kg = ks[:, g, :]  # [ctx, head_dim]
            vg = vs[:, g, :]
            scores = scale * (q[s, h].float() @ kg.t())  # [ctx]
            w = torch.softmax(scores, dim=0)
            o_ref[s, h] = w @ vg
            for j in range(ctx):
                phys = seq_phys[s][j // page_size]
                off = j % page_size
                mass_ref[phys, off, h] += w[j] * ctx  # scaled by ctx, own column for h
    return o_ref, mass_ref


def _check_decode(head_cfg, page_size, context_lens, dtype):
    q, k_pool, v_pool, page_tables, context_lens_t, seq_phys, ref_k, ref_v = (
        _build_paged(head_cfg, page_size, context_lens, dtype, seed=1)
    )
    total_pages = k_pool.size(0)
    head_dim = head_cfg[2]
    scale = 1.0 / (head_dim**0.5)

    # ops.attn_decode runs the tensor-core kernel for fp16/bf16, scalar for fp32.
    # Mass is per query head (head_cfg[0]); each head writes its own column.
    mass_pool = torch.zeros(
        total_pages, page_size, head_cfg[0], device="cuda", dtype=torch.float32
    )
    # decay=1.0 disables the EMA gain, so mass accumulates the raw normalized
    # weight the reference computes.
    decay = torch.ones(len(context_lens), device="cuda", dtype=torch.float32)
    o = ops.attn_decode(
        q, k_pool, v_pool, mass_pool, page_tables, context_lens_t, scale, decay,
    )

    # Scalar oracle op on the same inputs (fresh mass pool).
    mass_scalar = torch.zeros_like(mass_pool)
    o_scalar = torch.ops.pulsar.attn_decode_scalar(
        q, k_pool, v_pool, mass_scalar, page_tables, context_lens_t, scale, decay,
    )

    o_ref, mass_ref = _reference(
        head_cfg,
        page_size,
        context_lens,
        q,
        seq_phys,
        ref_k,
        ref_v,
        total_pages,
        scale,
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
    # No slot may differ beyond fp32 accumulation noise; this catches a mass
    # value landing on the wrong (page, offset, head h).
    m_sc_ok = torch.allclose(mass_pool, mass_scalar, atol=1e-4, rtol=1e-4)

    # Forced-split (flash-decode) op at several split counts, each checked
    # against the reference and the scalar oracle. For fp32 the split op routes
    # to the scalar kernel (no tensor-core path), so it trivially matches.
    split_m_vs_scalar = 0.0
    for num_splits in SPLIT_COUNTS:
        mass_split = torch.zeros_like(mass_pool)
        o_split = torch.ops.pulsar.attn_decode_split(
            q,
            k_pool,
            v_pool,
            mass_split,
            page_tables,
            context_lens_t,
            scale,
            num_splits,
            decay,
        )
        o_ok = o_ok and torch.allclose(
            o_split.float(), o_ref, atol=tols["atol"], rtol=tols["rtol"]
        )
        m_ok = m_ok and torch.allclose(mass_split, mass_ref, atol=1e-3, rtol=1e-3)
        o_sc_ok = o_sc_ok and torch.allclose(
            o_split.float(), o_scalar.float(), atol=tols["atol"], rtol=tols["rtol"]
        )
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


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


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
        page_size = 16
        q = torch.randn(3, 4, 8, device="cuda", dtype=dtype)
        k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        mass_pool = torch.zeros(8, page_size, 4, device="cuda", dtype=torch.float32)
        page_tables = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=torch.int32, device="cuda"
        )
        context_lens = torch.tensor([5, 20, 33], dtype=torch.int32, device="cuda")
        decay = torch.ones(3, dtype=torch.float32, device="cuda")
        torch.library.opcheck(
            torch.ops.pulsar.attn_decode,
            (q, k_pool, v_pool, mass_pool, page_tables, context_lens, 0.35, decay),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_decode_split():
    for dtype in DTYPES:
        page_size = 16
        q = torch.randn(3, 4, 8, device="cuda", dtype=dtype)
        k_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        v_pool = torch.randn(8, page_size, 2, 8, device="cuda", dtype=dtype)
        mass_pool = torch.zeros(8, page_size, 4, device="cuda", dtype=torch.float32)
        page_tables = torch.tensor(
            [[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=torch.int32, device="cuda"
        )
        context_lens = torch.tensor([5, 20, 33], dtype=torch.int32, device="cuda")
        decay = torch.ones(3, dtype=torch.float32, device="cuda")
        torch.library.opcheck(
            torch.ops.pulsar.attn_decode_split,
            (q, k_pool, v_pool, mass_pool, page_tables, context_lens, 0.35, 2, decay),
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
