import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from pulsar.core.engine import Engine
from pulsar.rt.sync import Box, Mutex, PoisonedError


def _step_and_poison(
    box: Box[Engine], n: int
) -> tuple[list[int], list[int], list[bool]]:
    """Runs on the executor thread. Poisons the box right where the failure
    happened, not wherever the awaiting coroutine next resumes."""
    try:
        return box.get().step(n)
    except Exception as exc:
        box.poison(exc)
        raise


class ResultStream(AsyncIterator[int]):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[int] = asyncio.Queue()
        self._exc: BaseException | None = None

    def __aiter__(self) -> "ResultStream":
        return self

    async def __anext__(self) -> int:
        try:
            return await self._queue.get()
        except asyncio.QueueShutDown:
            if self._exc is not None:
                raise PoisonedError() from self._exc
            raise StopAsyncIteration


class DecodeScheduler:
    def __init__(
        self, engine: Mutex[Box[Engine]], *, decode_chunk_size: int = 1
    ) -> None:
        self._engine = engine
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._decode_chunk_size = decode_chunk_size
        self._streams: dict[int, ResultStream] = {}
        self._need_step = asyncio.Event()

        self._scheduled_task: asyncio.Task[None] = asyncio.create_task(self._run())

    def schedule(self, seq_id: int) -> ResultStream:
        if seq_id in self._streams:
            raise RuntimeError(
                f"session {seq_id} already has a turn in flight; "
                "a send() must finish before the next one starts"
            )

        stream = ResultStream()
        self._streams[seq_id] = stream

        self._need_step.set()

        return stream

    def discard(self, seq_id: int) -> ResultStream | None:
        if (response := self._streams.pop(seq_id, None)) is not None:
            response._queue.shutdown()
        return response

    async def _step(self) -> tuple[list[int], list[int], list[bool]]:
        async with self._engine.lock() as box:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, _step_and_poison, box, self._decode_chunk_size
            )

    async def _run(self) -> None:
        try:
            while True:
                await self._need_step.wait()
                self._need_step.clear()
                while self._streams:
                    for seq_id, token, done in zip(*await asyncio.shield(self._step())):
                        stream = self._streams[seq_id]
                        stream._queue.put_nowait(token)
                        if done:
                            stream._queue.shutdown(immediate=False)
                            del self._streams[seq_id]
        except Exception as exc:
            # Catching BaseException here would swallow a CancelledError as if
            # this task completed normally, instead of letting it propagate as
            # a real cancellation.
            for pending in self._streams.values():
                pending._exc = exc
                pending._queue.shutdown(immediate=False)
            self._streams.clear()

    async def aclose(self) -> None:
        self._scheduled_task.cancel()
        self._executor.shutdown(wait=False)
