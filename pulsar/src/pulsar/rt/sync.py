import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class Mutex[T]:
    def __init__(self, value: T) -> None:
        self._value = value
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def lock(self) -> AsyncIterator[T]:
        async with self._lock:
            yield self._value


class PoisonedError(Exception):
    """Raised by Box.get() once the boxed value has been poisoned."""


class Box[T]:
    """Holds a value that can be poisoned; get() refuses to hand out a poisoned one."""

    def __init__(self, value: T) -> None:
        self._value = value
        self._poison: BaseException | None = None

    def get(self) -> T:
        if self._poison is not None:
            raise PoisonedError() from self._poison
        return self._value

    def poison(self, exc: BaseException) -> None:
        self._poison = exc
