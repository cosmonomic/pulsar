"""Streaming ids back to text is pulsar.decode's job; see test_decode.py."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer

from pulsar.checkpoint import iter_data_paths

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


def test_encode_known_ids(tok):
    assert tok.encode("Hello, world!").ids == [9707, 11, 1879, 0]
    assert tok.encode("The quick brown fox jumps over 42 lazy dogs.").ids == [
        785,
        3974,
        13876,
        38835,
        34208,
        916,
        220,
        19,
        17,
        15678,
        12590,
        13,
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Hello, world!",
        "def foo(x):\n    return x*2  # comment",
        "The quick brown fox jumps over 42 lazy dogs.",
    ],
)
def test_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_chat_markers_are_special_ids(tok):
    ids = tok.encode("<|im_start|>user\nhi<|im_end|>\n").ids
    assert ids[0] == 151644  # <|im_start|>
    assert 151645 in ids  # <|im_end|>
