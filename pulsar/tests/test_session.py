"""Multi-turn Session over the C++ engine.

A tiny random model exercises the session mechanics, budget-ended turns and
KV continuation across turns via feed(), not chat quality. Session speaks
token ids only, so no tokenizer or text framing is involved here.
"""

import asyncio

import pytest
import torch

from pulsar.checkpoint import PulsarModelConfig
from pulsar.rt import Pulsar
from pulsar.rt.session import Session, SessionState

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the pulsar kernels"
)

MAX_TOKENS = 5
PAGE_SIZE = 16
DEVICE = "cuda"


def _tiny_config() -> PulsarModelConfig:
    """Fast, minimal Qwen2 shape for exercising the engine; not a meaningful model."""
    return PulsarModelConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        eos_id=-1,  # -1 disables EOS, so only a turn's own budget ends it
    )


def _random_weights(cfg: PulsarModelConfig) -> dict[str, torch.Tensor]:
    """Random CPU weights, keyed exactly as Qwen2Model expects (qwen2.cpp)."""
    q_dim = cfg.n_heads * cfg.head_dim
    kv_dim = cfg.n_kv_heads * cfg.head_dim

    def w(*shape: int) -> torch.Tensor:
        return torch.randn(*shape) * 0.02

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": w(cfg.vocab_size, cfg.hidden_size),
        "model.norm.weight": torch.ones(cfg.hidden_size),
    }
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        weights.update(
            {
                p + "input_layernorm.weight": torch.ones(cfg.hidden_size),
                p + "post_attention_layernorm.weight": torch.ones(cfg.hidden_size),
                p + "self_attn.q_proj.weight": w(q_dim, cfg.hidden_size),
                p + "self_attn.q_proj.bias": torch.zeros(q_dim),
                p + "self_attn.k_proj.weight": w(kv_dim, cfg.hidden_size),
                p + "self_attn.k_proj.bias": torch.zeros(kv_dim),
                p + "self_attn.v_proj.weight": w(kv_dim, cfg.hidden_size),
                p + "self_attn.v_proj.bias": torch.zeros(kv_dim),
                p + "self_attn.o_proj.weight": w(cfg.hidden_size, q_dim),
                p + "mlp.gate_proj.weight": w(cfg.intermediate_size, cfg.hidden_size),
                p + "mlp.up_proj.weight": w(cfg.intermediate_size, cfg.hidden_size),
                p + "mlp.down_proj.weight": w(cfg.hidden_size, cfg.intermediate_size),
            }
        )
    return weights


class _StubTokenizer:
    """Pulsar stores a tokenizer for callers' convenience but never calls it
    itself; this test feeds raw token ids directly."""


def _pulsar(cfg, weights, *, active_buffer_size):
    dev_weights = {
        k: v.to(device=DEVICE, dtype=torch.float32) for k, v in weights.items()
    }
    return Pulsar(
        dev_weights,
        {
            "model": "qwen2",
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "hidden": cfg.hidden_size,
            "intermediate": cfg.intermediate_size,
            "vocab": cfg.vocab_size,
            "rope_theta": float(cfg.rope_theta),
            "rms_eps": float(cfg.rms_norm_eps),
            "active_buffer_size": active_buffer_size,
            "page_size": PAGE_SIZE,
            "device": DEVICE,
            # Parked sessions hold KV, so the pool must cover running plus parked;
            # the constructor asserts max_sessions >= max_running.
            "max_sessions": 8,
            "max_running": 8,
            "max_chunk_size": 256,
            "eos_id": cfg.eos_id,
        },
        _StubTokenizer(),
        dtype=torch.float32,
    )


async def _turn(session: Session, tokens: list[int], max_tokens: int) -> list[int]:
    response = await session.send(tokens, max_tokens)
    return [tok async for tok in response]


async def _run_multi_turn() -> tuple[list[int], list[int], SessionState, SessionState]:
    cfg = _tiny_config()
    weights = _random_weights(cfg)
    pulsar = _pulsar(cfg, weights, active_buffer_size=16 * PAGE_SIZE)
    session = await pulsar.create_session([1, 2, 3])  # any non-empty prefix works here

    out1 = await _turn(session, [4, 5], MAX_TOKENS)
    state1 = await session.get_state()  # turn budget ended; parked with KV kept active
    out2 = await _turn(session, [6, 7], MAX_TOKENS)
    state2 = await session.get_state()  # turn 2 continues on the KV feed() re-admitted

    await pulsar.aclose()
    return out1, out2, state1, state2


def test_session_multi_turn_streams_and_continues():
    torch.manual_seed(0)
    out1, out2, state1, state2 = asyncio.run(_run_multi_turn())

    assert state1 == "parked"
    assert state2 == "parked"
    # eos_id -1, so each turn emits exactly its MAX_TOKENS budget; the second
    # turn generating at all proves feed() continuation.
    assert len(out1) == MAX_TOKENS
    assert len(out2) == MAX_TOKENS
