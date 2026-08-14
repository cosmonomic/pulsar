"""Correctness checks for the cache-form fused causal attention CUDA op.

torch.ops.pulsar.attn_causal_cache reads fixed-capacity KV buffers
[kv_heads, max_len, head_dim] bounded by a scalar cur_len and mutates a
[kv_heads, max_len] attention_mass buffer in place. These tests check:
  1. equivalence to the exact-length attn_causal when max_len == L;
  2. fixed-buffer bounding: garbage rows beyond valid_len do not affect results
     and mass beyond valid_len is untouched;
  3. torch.library.opcheck validates the Tensor(a!) mutation and schema.
Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

CONFIGS = [
    # (n_q_heads, n_kv_heads, seq_q, cur_len, head_dim)
    (4, 4, 1, 15, 64),  # MHA decode
    (4, 4, 5, 11, 64),  # MHA prefill
    (8, 2, 1, 127, 64),  # GQA decode
    (8, 2, 6, 122, 128),  # GQA prefill, head_dim 128
    (16, 16, 3, 197, 128),  # MHA, head_dim 128
    (12, 4, 1, 511, 64),  # GQA decode, longer cache
    (8, 8, 7, 293, 128),  # MHA prefill, head_dim 128
]

DTYPES = [torch.float32, torch.bfloat16]


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-4, rtol=1e-4)


def _make_inputs(cfg, dtype, dev="cuda"):
    n_q_heads, n_kv_heads, seq_q, cur_len, head_dim = cfg
    L = cur_len + seq_q
    q = torch.randn(n_q_heads, seq_q, head_dim, device=dev, dtype=dtype)
    k = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
    v = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
    attention_mass = torch.randn(n_kv_heads, L, device=dev, dtype=torch.float32)
    scale = 1.0 / (head_dim**0.5)
    return q, k, v, attention_mass, scale, cur_len, L


def _check_equivalence(cfg, dtype):
    n_q_heads, n_kv_heads, seq_q, cur_len, head_dim = cfg
    torch.manual_seed(0)
    q, k, v, attention_mass, scale, cur_len, L = _make_inputs(cfg, dtype)

    # attn_causal's causal_offset argument is cur_len here; K/V are sized exactly to L.
    o_ref, mass_ref = ops.attn_causal(q, k, v, attention_mass, scale, cur_len)

    mass = attention_mass.clone()
    o = ops.attn_causal_cache(q, k, v, mass, scale, cur_len)

    tols = _tols(dtype)
    o_err = (o.float() - o_ref.float()).abs().max().item()
    m_err = (mass - mass_ref).abs().max().item()
    o_ok = torch.allclose(
        o.float(), o_ref.float(), atol=tols["atol"], rtol=tols["rtol"]
    )
    m_ok = torch.allclose(mass, mass_ref, atol=1e-3, rtol=1e-3)
    return o_ok and m_ok, o_err, m_err


def _check_fixed_buffer(cfg, dtype):
    n_q_heads, n_kv_heads, seq_q, cur_len, head_dim = cfg
    torch.manual_seed(1)
    q, k, v, attention_mass, scale, cur_len, L = _make_inputs(cfg, dtype)

    # Tight-buffer reference (max_len == L).
    mass_tight = attention_mass.clone()
    o_tight = ops.attn_causal_cache(q, k, v, mass_tight, scale, cur_len)

    # max_len = L + pad; rows [L, max_len) are garbage padding.
    pad = 37
    max_len = L + pad
    k_cache = torch.randn(n_kv_heads, max_len, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(n_kv_heads, max_len, head_dim, device="cuda", dtype=dtype)
    k_cache[:, :L] = k
    v_cache[:, :L] = v
    mass_pad = torch.randn(n_kv_heads, max_len, device="cuda", dtype=torch.float32)
    mass_pad[:, :L] = attention_mass
    mass_pad_orig_tail = mass_pad[:, L:].clone()

    o = ops.attn_causal_cache(q, k_cache, v_cache, mass_pad, scale, cur_len)

    tols = _tols(dtype)
    o_err = (o.float() - o_tight.float()).abs().max().item()
    m_err = (mass_pad[:, :L] - mass_tight).abs().max().item()
    tail_err = (mass_pad[:, L:] - mass_pad_orig_tail).abs().max().item()
    o_ok = torch.allclose(
        o.float(), o_tight.float(), atol=tols["atol"], rtol=tols["rtol"]
    )
    m_ok = torch.allclose(mass_pad[:, :L], mass_tight, atol=1e-3, rtol=1e-3)
    tail_ok = tail_err == 0.0  # untouched rows must be bit-identical
    return o_ok and m_ok and tail_ok, o_err, m_err, tail_err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_equivalence_to_existing_op(cfg, dtype):
    ok, o_err, m_err = _check_equivalence(cfg, dtype)
    assert ok, f"cfg={cfg} dtype={dtype} o_maxerr={o_err:.2e} mass_maxerr={m_err:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fixed_buffer_bounding(cfg, dtype):
    ok, o_err, m_err, tail_err = _check_fixed_buffer(cfg, dtype)
    assert ok, (
        f"cfg={cfg} dtype={dtype} o_maxerr={o_err:.2e} mass_maxerr={m_err:.2e} "
        f"tail_err={tail_err:.2e}"
    )


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck():
    for dtype in DTYPES:
        q = torch.randn(4, 1, 8, device="cuda", dtype=dtype)
        k_cache = torch.randn(2, 16, 8, device="cuda", dtype=dtype)
        v_cache = torch.randn(2, 16, 8, device="cuda", dtype=dtype)
        attention_mass = torch.zeros(2, 16, device="cuda", dtype=torch.float32)
        torch.library.opcheck(
            torch.ops.pulsar.attn_causal_cache,
            (q, k_cache, v_cache, attention_mass, 0.35, 10),
            test_utils=_OPCHECK_UTILS,
        )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True
    for dtype in DTYPES:
        for cfg in CONFIGS:
            eq_ok, o_err, m_err = _check_equivalence(cfg, dtype)
            fb_ok, fo_err, fm_err, tail_err = _check_fixed_buffer(cfg, dtype)
            ok = eq_ok and fb_ok
            all_pass &= ok
            print(
                f"[{'PASS' if ok else 'FAIL'}] {str(dtype):15s} cfg={cfg} "
                f"eq(o={o_err:.2e},m={m_err:.2e}) "
                f"fixed(o={fo_err:.2e},m={fm_err:.2e},tail={tail_err:.2e})"
            )

    try:
        test_opcheck()
        print("[PASS] opcheck (test_schema/test_faketensor/test_aot_dispatch_dynamic)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] opcheck: {exc}")
        all_pass = False

    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
