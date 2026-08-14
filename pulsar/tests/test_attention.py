"""Compares torch.ops.pulsar.attn_causal against a plain-PyTorch reference.
Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

CONFIGS = [
    # (n_q_heads, n_kv_heads, seq_q, L, head_dim)
    (4, 4, 1, 16, 64),  # MHA decode
    (4, 4, 5, 16, 64),  # MHA prefill
    (8, 2, 1, 128, 64),  # GQA decode
    (8, 2, 6, 128, 128),  # GQA prefill, head_dim 128
    (16, 16, 3, 200, 128),  # MHA, head_dim 128
    (12, 4, 1, 512, 64),  # GQA decode, longer L
    (8, 8, 7, 300, 128),  # MHA prefill, head_dim 128
]

DTYPES = [torch.float32, torch.bfloat16]


def reference(q, k, v, attention_mass, scale, causal_offset):
    n_q_heads, seq_q, head_dim = q.shape
    n_kv_heads, L, _ = k.shape
    group = n_q_heads // n_kv_heads

    qf = q.float()
    kf = k.float()
    vf = v.float()

    o = torch.zeros(n_q_heads, seq_q, head_dim, device=q.device, dtype=torch.float32)
    attention_mass_out = attention_mass.clone()

    for h in range(n_q_heads):
        g = h // group
        scores = scale * (qf[h] @ kf[g].transpose(0, 1))  # [seq_q, L]
        rows = torch.arange(seq_q, device=q.device).unsqueeze(1)
        cols = torch.arange(L, device=q.device).unsqueeze(0)
        disallowed = cols > (causal_offset + rows)
        scores = scores.masked_fill(disallowed, float("-inf"))
        p = torch.softmax(scores, dim=-1)  # [seq_q, L]
        o[h] = p @ vf[g]
        attention_mass_out[g] += p.sum(dim=0)

    return o, attention_mass_out


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_attention_matches_reference(cfg, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    n_q_heads, n_kv_heads, seq_q, L, head_dim = cfg
    torch.manual_seed(0)
    dev = "cuda"

    q = torch.randn(n_q_heads, seq_q, head_dim, device=dev, dtype=dtype)
    k = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
    v = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
    attention_mass = torch.randn(n_kv_heads, L, device=dev, dtype=torch.float32)
    scale = 1.0 / (head_dim**0.5)
    causal_offset = L - seq_q  # the seq_q new tokens are the most recent keys

    o, imp = ops.attn_causal(q, k, v, attention_mass, scale, causal_offset)
    o_ref, imp_ref = reference(q, k, v, attention_mass, scale, causal_offset)

    tols = _tols(dtype)
    assert torch.allclose(o.float(), o_ref, atol=tols["atol"], rtol=tols["rtol"]), (
        f"o mismatch cfg={cfg} dtype={dtype} "
        f"max={(o.float() - o_ref).abs().max().item()}"
    )
    assert torch.allclose(imp, imp_ref, atol=1e-3, rtol=1e-3), (
        f"attention_mass mismatch cfg={cfg} dtype={dtype} "
        f"max={(imp - imp_ref).abs().max().item()}"
    )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True
    for dtype in DTYPES:
        for cfg in CONFIGS:
            n_q_heads, n_kv_heads, seq_q, L, head_dim = cfg
            torch.manual_seed(0)
            dev = "cuda"
            q = torch.randn(n_q_heads, seq_q, head_dim, device=dev, dtype=dtype)
            k = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
            v = torch.randn(n_kv_heads, L, head_dim, device=dev, dtype=dtype)
            attention_mass = torch.randn(n_kv_heads, L, device=dev, dtype=torch.float32)
            scale = 1.0 / (head_dim**0.5)
            causal_offset = L - seq_q
            o, imp = ops.attn_causal(q, k, v, attention_mass, scale, causal_offset)
            o_ref, imp_ref = reference(q, k, v, attention_mass, scale, causal_offset)
            tols = _tols(dtype)
            o_ok = torch.allclose(
                o.float(), o_ref, atol=tols["atol"], rtol=tols["rtol"]
            )
            i_ok = torch.allclose(imp, imp_ref, atol=1e-3, rtol=1e-3)
            o_err = (o.float() - o_ref).abs().max().item()
            i_err = (imp - imp_ref).abs().max().item()
            ok = o_ok and i_ok
            all_pass &= ok
            print(
                f"[{'PASS' if ok else 'FAIL'}] {str(dtype):15s} cfg={cfg} "
                f"o_maxerr={o_err:.2e} imp_maxerr={i_err:.2e}"
            )
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
