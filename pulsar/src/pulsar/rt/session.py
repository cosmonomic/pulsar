from typing import Literal

from pulsar.core.engine import Engine
from pulsar.rt.pages_view import PagesView
from pulsar.rt.streaming import DecodeScheduler, ResultStream
from pulsar.rt.sync import Box, Mutex

SessionState = Literal["active", "parked", "unknown"]


class Session:
    def __init__(
        self, seq_id: int, engine: Mutex[Box[Engine]], scheduler: DecodeScheduler
    ) -> None:
        self._seq_id = seq_id
        self._engine = engine
        self._scheduler = scheduler

    @property
    def seq_id(self) -> int:
        return self._seq_id

    async def send(self, tokens: list[int]):
        async with self._engine.lock() as box:
            box.get().feed(self.seq_id, tokens)

    async def recv(self) -> ResultStream:
        return self._scheduler.schedule(self.seq_id)

    async def get_stats(self) -> dict[str, int | list[int] | list[float]]:
        """Evict/recall accounting for this session since the last per-run reset."""
        async with self._engine.lock() as box:
            return box.get().stats(self.seq_id)

    async def get_context(self) -> PagesView:
        """The active pages the attention kernels see (recall on or off)."""
        async with self._engine.lock() as box:
            e = box.get()
            return PagesView(
                e.context_page_ids(self.seq_id), e.context_pages(self.seq_id)
            )

    async def get_history(self) -> PagesView:
        """The whole conversation merged across every KV component (requires recall)."""
        async with self._engine.lock() as box:
            e = box.get()
            return PagesView(
                e.history_page_ids(self.seq_id), e.history_pages(self.seq_id)
            )

    async def get_demoted_tokens_count(self) -> int:
        async with self._engine.lock() as box:
            return box.get().demoted_tokens_count(self.seq_id)

    async def get_state(self) -> SessionState:
        async with self._engine.lock() as box:
            return box.get().session_state(self.seq_id)

    async def destroy(self) -> None:
        self._scheduler.discard(self.seq_id)

        async with self._engine.lock() as box:
            box.get().release(self.seq_id)
