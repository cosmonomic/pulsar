import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from pulsar.checkpoint import iter_data_paths
from pulsar.decode import StreamingDecoder

_TOK_RELPATH = Path("qwen2.5-1.5b-instruct") / "tokenizer.json"
TOK_JSON = next(
    (p for base in iter_data_paths() if (p := base / _TOK_RELPATH).exists()),
    next(iter_data_paths()) / _TOK_RELPATH,
)

pytestmark = pytest.mark.skipif(
    not TOK_JSON.exists(), reason="qwen2.5 tokenizer.json not present"
)


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.from_file(str(TOK_JSON))


async def _ids(token_ids: Iterable[int]) -> AsyncIterator[int]:
    for token_id in token_ids:
        yield token_id


async def _decode(sd: StreamingDecoder, token_ids: Iterable[int]) -> str:
    return "".join([text async for text in sd.decode(_ids(token_ids))])


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "café 日本語 🚀 x",
        "def foo(x):\n    return x*2  # comment",
    ],
)
def test_streaming_roundtrip(tok, text):
    sd = StreamingDecoder(tok)
    joined = asyncio.run(_decode(sd, tok.encode(text).ids))
    assert joined == text


def test_special_tokens_emit_literal_text(tok):
    sd = StreamingDecoder(tok)
    im_start = tok.encode("<|im_start|>").ids[0]
    im_end = tok.encode("<|im_end|>").ids[0]
    assert asyncio.run(_decode(sd, [im_start])) == "<|im_start|>"
    assert asyncio.run(_decode(sd, [im_end])) == "<|im_end|>"
