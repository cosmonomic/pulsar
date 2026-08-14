import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI

from pulsar import checkpoint
from pulsar.rt import Pulsar
from pulsar.server.openai import ServerState
from pulsar.server.openai import router as openai_router

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_PAGE_SIZE = 16  # the tensor-core paged attention kernel requires 16 or 32


@dataclass
class Settings:
    model: str
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    active_buffer_size: int = 32768
    max_chunk_size: int = 256
    max_sessions: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.environ.get("PULSAR_MODEL")
        if not model:
            raise RuntimeError("PULSAR_MODEL must be set")
        return cls(
            model=model,
            device=os.environ.get("PULSAR_DEVICE", cls.device),
            dtype=_DTYPES[os.environ.get("PULSAR_DTYPE", "bfloat16")],
            active_buffer_size=int(
                os.environ.get("PULSAR_ACTIVE_BUFFER_SIZE", cls.active_buffer_size)
            ),
            max_chunk_size=int(
                os.environ.get("PULSAR_MAX_CHUNK_SIZE", cls.max_chunk_size)
            ),
            max_sessions=int(os.environ.get("PULSAR_MAX_SESSIONS", cls.max_sessions)),
        )


def _load_pulsar(settings: Settings) -> Pulsar:
    model = checkpoint.load(
        checkpoint.model_path(settings.model),
        device=settings.device,
        dtype=settings.dtype,
    )
    cfg = model.config
    return Pulsar(
        model.weights,
        {
            "model": settings.model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "hidden": cfg.hidden_size,
            "intermediate": cfg.intermediate_size,
            "vocab": cfg.vocab_size,
            "rope_theta": float(cfg.rope_theta),
            "rms_eps": float(cfg.rms_norm_eps),
            "active_buffer_size": settings.active_buffer_size,
            "page_size": _PAGE_SIZE,
            "device": settings.device,
            "max_sessions": settings.max_sessions,
            "max_running": settings.max_sessions,
            "max_chunk_size": settings.max_chunk_size,
            "eos_id": cfg.eos_id,
        },
        model.tokenizer,
        dtype=settings.dtype,
    )


def build_app(build_state: Callable[[], ServerState]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = build_state()
        app.state.server = state
        try:
            yield
        finally:
            await state.pulsar.aclose()

    app = FastAPI(lifespan=lifespan)
    app.include_router(openai_router, prefix="/api/openai/v1")
    return app


def create_app() -> FastAPI:
    def build_state() -> ServerState:
        settings = Settings.from_env()
        return ServerState(_load_pulsar(settings), settings.model, settings.max_sessions)

    return build_app(build_state)


app = create_app()
