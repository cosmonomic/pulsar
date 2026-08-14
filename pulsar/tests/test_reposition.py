"""Unit tests for the reposition_kv CUDA op.

The paged KV pool stores ROPE'd keys. Moving a key to a new pos is one
extra rotation by the delta angle, so reposition must compose with rope:
    reposition_kv(rope(raw, p_old), p_old -> p_new) == rope(raw, p_new).

Each test ropes a random raw K at old pos, writes it into a pool at
scattered slots, repositions to new pos, and compares against a direct
rope(raw, p_new). Covers multiple heads, head_dim {64, 128}, scattered slots,
forward/backward/no-op moves, and fp32/fp16/bf16.

Requires a CUDA device and the built pulsar extension.
"""

import pytest
import torch

from pulsar.core import ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
THETA = 1_000_000.0


def _rope_rows(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """RoPE a [M, n_kv_heads, head_dim] tensor at the given per-row pos."""
    roped = ops.rope(x.transpose(0, 1).contiguous(), pos, THETA)
    return roped.transpose(0, 1).contiguous()


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 3e-2, 3e-2),
    ],
)
def test_reposition_matches_rope(head_dim, dtype, atol, rtol):
    torch.manual_seed(0)
    n_kv = 3
    num_pages, page_size = 8, 16
    total = num_pages * page_size
    M = 24

    # Scattered, distinct physical slots.
    slots = torch.randperm(total, device=DEV)[:M].to(torch.int32)

    # Old/new pos covering forward, backward, and no-op moves.
    old_pos = torch.randint(0, 300, (M,), device=DEV)
    new_pos = torch.randint(0, 300, (M,), device=DEV)
    old_pos[0], new_pos[0] = 123, 123  # no-op
    old_pos[1], new_pos[1] = 100, 180  # forward (new > old)
    old_pos[2], new_pos[2] = 200, 40  # backward (new < old)
    old_pos_i64 = old_pos.to(torch.int64)
    new_pos_i64 = new_pos.to(torch.int64)

    raw = torch.randn(M, n_kv, head_dim, device=DEV, dtype=dtype)
    roped_old = _rope_rows(raw, old_pos_i64)

    pool = torch.zeros(num_pages, page_size, n_kv, head_dim, device=DEV, dtype=dtype)
    pool.view(total, n_kv, head_dim)[slots.to(torch.int64)] = roped_old

    ops.reposition_kv(
        pool, slots, old_pos.to(torch.int32), new_pos.to(torch.int32), THETA
    )

    got = pool.view(total, n_kv, head_dim)[slots.to(torch.int64)]
    expected = _rope_rows(raw, new_pos_i64)
    torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)


def test_reposition_noop_leaves_keys_unchanged():
    # Every key uses new == old; the pool must stay bit-identical afterwards.
    torch.manual_seed(1)
    n_kv, head_dim = 2, 64
    num_pages, page_size = 4, 16
    total = num_pages * page_size
    M = 10

    slots = torch.randperm(total, device=DEV)[:M].to(torch.int32)
    pos = torch.randint(0, 200, (M,), device=DEV)
    raw = torch.randn(M, n_kv, head_dim, device=DEV)
    roped = _rope_rows(raw, pos.to(torch.int64))

    pool = torch.zeros(num_pages, page_size, n_kv, head_dim, device=DEV)
    pool.view(total, n_kv, head_dim)[slots.to(torch.int64)] = roped
    before = pool.clone()

    ops.reposition_kv(pool, slots, pos.to(torch.int32), pos.to(torch.int32), THETA)

    assert torch.equal(pool, before)
