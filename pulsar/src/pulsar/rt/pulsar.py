from typing import Any

import torch
from tokenizers import Tokenizer

from pulsar.core.engine import Engine
from pulsar.rt.streaming import DecodeScheduler
from pulsar.rt.session import Session
from pulsar.rt.sync import Box, Mutex


class Pulsar:
    """Everything scoped to one sequence is on the Session create_session() returns."""

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        config: dict[str, Any],
        tokenizer: Tokenizer,
        *,
        dtype: torch.dtype,
        decode_chunk_size: int = 1,
    ) -> None:
        self._engine: Mutex[Box[Engine]] = Mutex(Box(Engine(weights, config, dtype)))
        self._scheduler = DecodeScheduler(
            self._engine, decode_chunk_size=decode_chunk_size
        )

        self.tokenizer = tokenizer

    async def create_session(
        self,
        sink: list[int],
        *,
        overrides: dict[str, Any] | None = None,
    ) -> Session:
        async with self._engine.lock() as box:
            seq_id = box.get().create_session(sink, overrides or {})

        return Session(seq_id, self._engine, self._scheduler)

    async def aclose(self) -> None:
        await self._scheduler.aclose()
