"""Correctness checks for the Marlin-style w4a16 dequant-GEMM CUDA ops.

torch.ops.pulsar.repack_w4a16 transforms the simple compressed-tensors int4
packing into an opaque Marlin fragment layout; torch.ops.pulsar.gemm_w4a16_marlin
then computes y = x @ dequant(W)^T from it, matching gemm_w4a16's result. These
tests validate the pair against a plain-PyTorch dequant + fp32 matmul oracle
(the same one test_w4a16_gemm.py uses) and cross-check it against gemm_w4a16 on
the same logical weights. Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

DTYPES = [torch.float16, torch.bfloat16]

# (N, K)
NK_SHAPES = [(128, 128), (256, 512), (512, 4096), (4096, 4096)]

M_SIZES = [1, 4, 16, 64]

GROUP_SIZES = [64, 128]


def _tol(dtype):
    return 2e-2 if dtype == torch.bfloat16 else 5e-3


def _pack_int4(vals: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values [N, K] in [-8, 7] into int32 [N, K/8]."""
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
    scale_expanded = scale.repeat_interleave(group_size, dim=1)  # [N, K]
    w_deq = vals.float() * scale_expanded.float()  # [N, K]
    return x.float() @ w_deq.t()  # [M, N]


def _run(m, n, k, group_size, dtype, seed=0):
    torch.manual_seed(seed)
    vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)  # [-8, 7]
    scale = torch.rand(n, k // group_size) * 0.04 + 0.01  # positive
    x = torch.randn(m, k) * 0.1

    weight_packed = _pack_int4(vals).cuda()
    weight_scale = scale.to(dtype).cuda()
    x_dev = x.to(dtype).cuda()

    weight_marlin, scale_marlin = ops.repack_w4a16(
        weight_packed, weight_scale, group_size
    )
    y = ops.gemm_w4a16_marlin(x_dev, weight_marlin, scale_marlin, group_size)

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
def test_gemm_w4a16_marlin(m, nk, group_size, dtype):
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
@pytest.mark.parametrize("nk", [(256, 512), (512, 4096)])
@pytest.mark.parametrize("m", [1, 16, 64])
def test_marlin_matches_gemm_w4a16(m, nk, group_size, dtype):
    # Same logical weights through gemm_w4a16 and through repack+marlin must give
    # matching y (within dtype tolerance vs the shared fp32 oracle).
    n, k = nk
    torch.manual_seed(1)
    vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
    scale = torch.rand(n, k // group_size) * 0.04 + 0.01
    x = torch.randn(m, k) * 0.1
    weight_packed = _pack_int4(vals).cuda()
    weight_scale = scale.to(dtype).cuda()
    x_dev = x.to(dtype).cuda()

    y_simple = ops.gemm_w4a16(x_dev, weight_packed, weight_scale, group_size)
    weight_marlin, scale_marlin = ops.repack_w4a16(
        weight_packed, weight_scale, group_size
    )
    y_marlin = ops.gemm_w4a16_marlin(x_dev, weight_marlin, scale_marlin, group_size)

    denom = y_simple.float().abs().max().item()
    diff = (y_marlin.float() - y_simple.float()).abs().max().item()
    rel = diff / denom if denom > 0 else diff
    assert rel <= _tol(dtype), (
        f"m={m} n={n} k={k} group_size={group_size} dtype={dtype} "
        f"marlin-vs-simple rel={rel:.3e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_repack_roundtrip():
    # Unpacking weight_marlin back through the documented nibble map must recover
    # the original signed int4 weights (padding aside).
    n, k, group_size = 128, 256, 64
    block_n, block_k, k_tile, n_sub = 64, 64, 32, 8
    torch.manual_seed(2)
    vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
    scale = torch.rand(n, k // group_size) * 0.04 + 0.01
    weight_packed = _pack_int4(vals).cuda()
    weight_scale = scale.to(torch.float32).cuda()

    weight_marlin, scale_marlin = ops.repack_w4a16(
        weight_packed, weight_scale, group_size
    )

    # scale_marlin is the transpose of weight_scale.
    assert torch.allclose(scale_marlin.cpu(), scale.t().float(), atol=1e-6)

    wm = weight_marlin.cpu().to(torch.int64) & 0xFFFFFFFF
    recovered = torch.zeros(n, k, dtype=torch.int32)
    n_padded = -(-n // block_n) * block_n
    k_padded = -(-k // block_k) * block_k
    n_chunks = k_padded // block_k
    k_tiles_per_chunk = block_k // k_tile

    def nibble_to_k_offset(nibble_idx, thread_group):
        base = [0, 8, 16, 24, 0, 8, 16, 24][nibble_idx]
        return base + thread_group * 2 + (1 if nibble_idx >= 4 else 0)

    for col_block in range(n_padded // block_n):
        for k_chunk in range(n_chunks):
            for k_tile_idx in range(k_tiles_per_chunk):
                for sub_idx in range(n_sub):
                    for lane in range(32):
                        word_idx = (
                            (col_block * n_chunks + k_chunk) * k_tiles_per_chunk
                            + k_tile_idx
                        ) * n_sub + sub_idx
                        word_idx = word_idx * 32 + lane
                        word = int(wm[word_idx])
                        col = (col_block * n_sub + sub_idx) * 8 + (lane >> 2)
                        thread_group = lane & 3
                        k_base = (k_chunk * k_tiles_per_chunk + k_tile_idx) * k_tile
                        for nibble_idx in range(8):
                            nib = (word >> (4 * nibble_idx)) & 0xF
                            val = nib ^ 8  # undo sign-bit flip
                            val = val - 16 if val >= 8 else val
                            k_idx = k_base + nibble_to_k_offset(
                                nibble_idx, thread_group
                            )
                            if col < n and k_idx < k:
                                recovered[col, k_idx] = val
    assert torch.equal(recovered, vals), "weight_marlin did not round-trip"


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_repack_w4a16():
    for dtype in DTYPES:
        n, k, group_size = 128, 128, 64
        vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
        scale = torch.rand(n, k // group_size) * 0.04 + 0.01
        weight_packed = _pack_int4(vals).cuda()
        weight_scale = scale.to(dtype).cuda()
        torch.library.opcheck(
            torch.ops.pulsar.repack_w4a16,
            (weight_packed, weight_scale, group_size),
            test_utils=_OPCHECK_UTILS,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_opcheck_gemm_w4a16_marlin():
    for dtype in DTYPES:
        n, k, group_size = 128, 128, 64
        vals = torch.randint(-8, 8, (n, k), dtype=torch.int32)
        scale = torch.rand(n, k // group_size) * 0.04 + 0.01
        weight_packed = _pack_int4(vals).cuda()
        weight_scale = scale.to(dtype).cuda()
        weight_marlin, scale_marlin = ops.repack_w4a16(
            weight_packed, weight_scale, group_size
        )
        x = (torch.randn(4, k) * 0.1).to(dtype).cuda()
        torch.library.opcheck(
            torch.ops.pulsar.gemm_w4a16_marlin,
            (x, weight_marlin, scale_marlin, group_size),
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
                        f"[{'PASS' if ok else 'FAIL'}] marlin "
                        f"{str(dtype):15s} m={m:3d} n={n:4d} k={k:5d} "
                        f"g={group_size:3d} abs={max_abs_err:.2e} "
                        f"rel={rel_err:.2e}"
                    )
    print("\nALL PASS" if all_pass else "\nSOME FAILED")
    return all_pass


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_standalone() else 1)
