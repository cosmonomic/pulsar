"""Correctness checks for the fused w4a16 dequant-GEMM CUDA op.

torch.ops.pulsar.gemm_w4a16 computes y = x @ dequant(W)^T for a Linear with
group-wise symmetric int4 weights: y[m, n] = sum_k x[m, k] * w_deq[n, k] with
w_deq[n, k] = int4(weight_packed[n, k]) * weight_scale[n, k // group_size].

The op reads the int4 weights packed from global memory and dequantizes them
only into an on-chip tensor-core tile (fp32 accumulate). These tests validate it
against a plain-PyTorch dequant + fp32 matmul oracle built under the exact
packing convention the kernel expects, so kernel and oracle stay self-consistent.
Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

DTYPES = [torch.float16, torch.bfloat16]

# (N, K)
NK_SHAPES = [(128, 128), (256, 512), (512, 4096)]

M_SIZES = [1, 4, 16, 64]

GROUP_SIZES = [64, 128]

# Forced K-split counts for the split op. 1 exercises the split+combine path at a
# single split; the larger counts, with the small-K shapes above (K=128 -> only 2
# BK=64 K-chunks), produce splits that exceed the chunk count, covering the
# empty-split path (extra splits contribute 0).
SPLIT_COUNTS = [1, 2, 3, 4, 8]


def _tol(dtype):
    # Relative to the output magnitude; only input rounding to the low-precision
    # dtype separates the kernel from the fp32 oracle (both accumulate in fp32).
    return 2e-2 if dtype == torch.bfloat16 else 5e-3


def _pack_int4(vals: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [N, K] in [-8, 7] into int32 [N, K/8].

    Column k lands in nibble k % 8 (bits [4*(k%8) .. +3]) of the int32 at column
    k // 8, LSB-first, two's-complement. Mirrors the kernel's unpack convention.
    """
    n, k = vals.shape
    assert k % 8 == 0
    v = vals.to(torch.int32) & 0xF
    packed = torch.zeros(n, k // 8, dtype=torch.int32)
    for j in range(8):
        packed |= v[:, j::8] << (4 * j)
    return packed


def _oracle(
    vals: torch.Tensor, scale: torch.Tensor, x: torch.Tensor, group_size: int
) -> torch.Tensor:
    """y = x @ (vals * scale_expanded)^T, computed in fp32."""
    scale_expanded = scale.repeat_interleave(group_size, dim=1)  # [N, K]
    w_deq = vals.float() * scale_expanded.float()  # [N, K]
    return x.float() @ w_deq.t()  # [M, N]


def _run(m, n, k, group_size, dtype, seed=0, num_splits=None):
    torch.manual_seed(seed)
    vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)  # [-8, 7]
    scale = torch.rand(n, k // group_size) * 0.04 + 0.01  # positive
    x = torch.randn(m, k) * 0.1

    weight_packed = _pack_int4(vals).cuda()
    weight_scale = scale.to(dtype).cuda()  # kernel upcasts to fp32 internally
    x_dev = x.to(dtype).cuda()

    if num_splits is None:
        y = ops.gemm_w4a16(x_dev, weight_packed, weight_scale, group_size)
    else:
        y = torch.ops.pulsar.gemm_w4a16_split(
            x_dev, weight_packed, weight_scale, group_size, num_splits
        )

    y_ref = _oracle(vals, scale, x, group_size).cuda()
    max_abs_err = (y.float() - y_ref).abs().max().item()
    denom = y_ref.abs().max().item()
    rel_err = max_abs_err / denom if denom > 0 else max_abs_err
    return y, y_ref, max_abs_err, rel_err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("nk", NK_SHAPES)
@pytest.mark.parametrize("m", M_SIZES)
def test_gemm_w4a16(m, nk, group_size, dtype):
    n, k = nk
    y, y_ref, max_abs_err, rel_err = _run(m, n, k, group_size, dtype)
    tol = _tol(dtype)
    assert rel_err <= tol, (
        f"m={m} n={n} k={k} group_size={group_size} dtype={dtype} "
        f"max_abs_err={max_abs_err:.3e} rel_err={rel_err:.3e} tol={tol:.1e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("nk", NK_SHAPES)
@pytest.mark.parametrize("m", M_SIZES)
@pytest.mark.parametrize("num_splits", SPLIT_COUNTS)
def test_gemm_w4a16_split(num_splits, m, nk, group_size, dtype):
    # The forced-split op must match the fp32 oracle to the same tolerance as the
    # auto op, including num_splits that exceed the K-chunk count (empty splits).
    n, k = nk
    y, y_ref, max_abs_err, rel_err = _run(
        m, n, k, group_size, dtype, num_splits=num_splits
    )
    tol = _tol(dtype)
    assert rel_err <= tol, (
        f"num_splits={num_splits} m={m} n={n} k={k} group_size={group_size} "
        f"dtype={dtype} max_abs_err={max_abs_err:.3e} rel_err={rel_err:.3e} "
        f"tol={tol:.1e}"
    )


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_gemm_w4a16():
    for dtype in DTYPES:
        n, k, group_size = 128, 128, 64
        vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
        scale = torch.rand(n, k // group_size) * 0.04 + 0.01
        weight_packed = _pack_int4(vals).cuda()
        weight_scale = scale.to(dtype).cuda()
        x = (torch.randn(4, k) * 0.1).to(dtype).cuda()
        torch.library.opcheck(
            torch.ops.pulsar.gemm_w4a16,
            (x, weight_packed, weight_scale, group_size),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_gemm_w4a16_split():
    for dtype in DTYPES:
        n, k, group_size = 128, 128, 64
        vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
        scale = torch.rand(n, k // group_size) * 0.04 + 0.01
        weight_packed = _pack_int4(vals).cuda()
        weight_scale = scale.to(dtype).cuda()
        x = (torch.randn(4, k) * 0.1).to(dtype).cuda()
        for num_splits in (1, 2, 4):
            torch.library.opcheck(
                torch.ops.pulsar.gemm_w4a16_split,
                (x, weight_packed, weight_scale, group_size, num_splits),
                test_utils=_OPCHECK_UTILS,
            )


def _run_standalone() -> bool:
    if not torch.cuda.is_available():
        print("CUDA required")
        return False
    all_pass = True
    for dtype in DTYPES:
        for group_size in GROUP_SIZES:
            for n, k in NK_SHAPES:
                for m in M_SIZES:
                    _, _, max_abs_err, rel_err = _run(m, n, k, group_size, dtype)
                    ok = rel_err <= _tol(dtype)
                    all_pass &= ok
                    print(
                        f"[{'PASS' if ok else 'FAIL'}] w4a16 "
                        f"{str(dtype):15s} m={m:3d} n={n:4d} k={k:5d} "
                        f"g={group_size:3d} abs={max_abs_err:.2e} "
                        f"rel={rel_err:.2e}"
                    )
    for dtype in DTYPES:
        for group_size in GROUP_SIZES:
            for n, k in NK_SHAPES:
                for m in M_SIZES:
                    for ns in SPLIT_COUNTS:
                        _, _, max_abs_err, rel_err = _run(
                            m, n, k, group_size, dtype, num_splits=ns
                        )
                        ok = rel_err <= _tol(dtype)
                        all_pass &= ok
                        print(
                            f"[{'PASS' if ok else 'FAIL'}] split "
                            f"{str(dtype):15s} m={m:3d} n={n:4d} k={k:5d} "
                            f"g={group_size:3d} ns={ns} abs={max_abs_err:.2e} "
                            f"rel={rel_err:.2e}"
                        )
    try:
        test_opcheck_gemm_w4a16()
        test_opcheck_gemm_w4a16_split()
        print("[PASS] opcheck gemm_w4a16 (schema/faketensor/aot_dispatch_dynamic)")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] opcheck gemm_w4a16: {exc}")
        all_pass = False
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
