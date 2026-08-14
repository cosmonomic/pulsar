"""Correctness checks for the RoPE CUDA op.

Compares torch.ops.pulsar.rope against a plain-PyTorch reference implementing the
HF/Qwen2 "rotate_half" convention over the full head_dim. Exercises head_dim
64/128, contiguous-with-offset and arbitrary/non-contiguous pos (the
re-RoPE case), theta 10000 and 1000000, fp32/bf16. Requires a CUDA device and
the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

CONFIGS = [
    # (n_heads, seq, head_dim)
    (4, 8, 64),
    (8, 16, 64),
    (16, 5, 128),
    (12, 32, 128),
    (2, 1, 64),  # single-row decode
    (32, 64, 128),
]

DTYPES = [torch.float32, torch.bfloat16]
THETAS = [10000.0, 1000000.0]


def _rotate_half(x):
    h = x.shape[-1] // 2
    x1 = x[..., :h]
    x2 = x[..., h:]
    return torch.cat((-x2, x1), dim=-1)


def reference(x, pos, theta):
    """Plain-PyTorch RoPE reference for the HF/Qwen2 rotate_half convention.

    inv_freq[j] = theta^(-2j/head_dim); a = pos * inv_freq; then
    out = x * cos(a) + rotate_half(x) * sin(a), with cos/sin tiled across the
    two halves:
      out[j]   = x[j]   * cos - x[j+h] * sin
      out[j+h] = x[j+h] * cos + x[j]   * sin

    The angle is formed in fp64 (pos can be large, and inv_freq's relative
    error scales into the angle by the pos), then cos/sin are rounded to
    fp32 and the rotation runs in fp32, matching the kernel's precision policy.
    """
    head_dim = x.shape[-1]
    xf = x.float()
    pos = pos.double()  # [seq]

    j = torch.arange(head_dim // 2, device=x.device, dtype=torch.float64)
    inv_freq = theta ** (-2.0 * j / head_dim)  # [h], fp64
    a = pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq, h], fp64
    a = torch.cat((a, a), dim=-1)  # [seq, head_dim], tiled across halves
    cos = a.cos().float().unsqueeze(0)  # [1, seq, head_dim], fp32
    sin = a.sin().float().unsqueeze(0)

    return xf * cos + _rotate_half(xf) * sin


def _tols(dtype):
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-4, rtol=1e-4)


def _make_positions(seq, kind, dev):
    if kind == "offset":
        offset = 137
        return torch.arange(offset, offset + seq, device=dev, dtype=torch.long)
    # arbitrary / non-contiguous, exercises re-RoPE under a recompacted layout
    g = torch.Generator(device="cpu").manual_seed(1234)
    p = torch.randint(0, 4096, (seq,), generator=g, dtype=torch.long)
    return p.to(dev)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("pos_kind", ["offset", "arbitrary"])
def test_rope_matches_reference(cfg, dtype, theta, pos_kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    n_heads, seq, head_dim = cfg
    torch.manual_seed(0)
    dev = "cuda"

    x = torch.randn(n_heads, seq, head_dim, device=dev, dtype=dtype)
    pos = _make_positions(seq, pos_kind, dev)

    out = ops.rope(x, pos, theta)
    ref = reference(x, pos, theta)

    tols = _tols(dtype)
    assert torch.allclose(out.float(), ref, atol=tols["atol"], rtol=tols["rtol"]), (
        f"rope mismatch cfg={cfg} dtype={dtype} theta={theta} pos={pos_kind} "
        f"max={(out.float() - ref).abs().max().item()}"
    )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True
    dev = "cuda"
    for dtype in DTYPES:
        for theta in THETAS:
            for pos_kind in ["offset", "arbitrary"]:
                for cfg in CONFIGS:
                    n_heads, seq, head_dim = cfg
                    torch.manual_seed(0)
                    x = torch.randn(n_heads, seq, head_dim, device=dev, dtype=dtype)
                    pos = _make_positions(seq, pos_kind, dev)
                    out = ops.rope(x, pos, theta)
                    ref = reference(x, pos, theta)
                    tols = _tols(dtype)
                    ok = torch.allclose(
                        out.float(), ref, atol=tols["atol"], rtol=tols["rtol"]
                    )
                    err = (out.float() - ref).abs().max().item()
                    all_pass &= ok
                    print(
                        f"[{'PASS' if ok else 'FAIL'}] {str(dtype):15s} "
                        f"theta={theta:>9.0f} pos={pos_kind:9s} cfg={cfg} "
                        f"maxerr={err:.2e}"
                    )
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
