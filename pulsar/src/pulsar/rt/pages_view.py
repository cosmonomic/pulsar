from collections.abc import Iterator
from itertools import chain


class _Keys:
    """Page indices, in address order."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __len__(self) -> int:
        return len(self._ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self._ids)


class _Values:
    """Per-page token-id lists, in address order."""

    def __init__(self, pages: list[list[int]]) -> None:
        self._pages = pages

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._pages)


class PagesView:
    """A sequence's pages, in address order."""

    def __init__(self, ids: list[int], pages: list[list[int]]) -> None:
        self.keys = _Keys(ids)
        self.values = _Values(pages)

    def tokens(self) -> Iterator[int]:
        return chain.from_iterable(self.values)
