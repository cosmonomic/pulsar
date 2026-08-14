"""ChatML turn formatting and generation-output parsing."""

import asyncio
from collections.abc import AsyncIterator
from typing import cast

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pulsar.conversation import ChatMLConversation, TruncatedError
from pulsar.rt.session import Session


def _tok() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"],
    )
    tok.train_from_iterator(["hello world foo bar baz qux"] * 200, trainer=trainer)
    return tok


class _FakeSession:
    """Just enough of Session's surface for ChatMLConversation.send()/recv()."""

    def __init__(self, reply_ids: list[int]) -> None:
        self.sent: list[list[int]] = []
        self._reply_ids = reply_ids

    async def send(self, tokens: list[int]) -> None:
        self.sent.append(tokens)

    async def recv(self) -> AsyncIterator[int]:
        async def _stream() -> AsyncIterator[int]:
            for token_id in self._reply_ids:
                yield token_id

        return _stream()


def test_turn_markers():
    assert ChatMLConversation.turn_open("user") == "<|im_start|>user\n"
    assert ChatMLConversation.turn_open("assistant") == "<|im_start|>assistant\n"
    assert ChatMLConversation.turn_open("system") == "<|im_start|>system\n"
    assert ChatMLConversation.turn_close() == "<|im_end|>\n"


def test_compose_turn():
    turn = ChatMLConversation.turn_open("user") + "hi" + ChatMLConversation.turn_close()
    assert turn == "<|im_start|>user\nhi<|im_end|>\n"


async def _collect(chunks: AsyncIterator[str]) -> str:
    return "".join([c async for c in chunks])


def test_send_decodes_and_stops_on_the_stop_marker():
    tok = _tok()
    im_end = tok.token_to_id("<|im_end|>")
    assert im_end is not None
    session = cast(Session, _FakeSession(tok.encode("hello world").ids + [im_end]))
    p = ChatMLConversation(tok, session)

    assert asyncio.run(_collect(p.recv())) == "hello world"


def test_send_raises_when_the_stream_has_no_stop_marker():
    tok = _tok()
    session = cast(Session, _FakeSession(tok.encode("hello world").ids))
    p = ChatMLConversation(tok, session)

    with pytest.raises(TruncatedError):
        asyncio.run(_collect(p.recv()))


def test_token_count_is_just_the_number_of_chunks_iterated():
    tok = _tok()
    im_end = tok.token_to_id("<|im_end|>")
    assert im_end is not None
    ids = tok.encode("hello world").ids
    session = cast(Session, _FakeSession([*ids, im_end]))
    p = ChatMLConversation(tok, session)

    async def run() -> int:
        count = 0
        async for _ in p.recv():
            count += 1
        return count

    # every decoded token yields exactly one chunk (possibly ""), including
    # the stop marker itself, so a caller counting iterations gets the token
    # count for free
    assert asyncio.run(run()) == len(ids) + 1


def test_send_opens_a_turn_before_the_content():
    tok = _tok()
    fake = _FakeSession([])
    p = ChatMLConversation(tok, cast(Session, fake))

    sent = asyncio.run(p.send("user", "hi"))

    open_ids = tok.encode(p.turn_open("user")).ids
    content_ids = tok.encode("hi").ids
    assert fake.sent == [open_ids, content_ids]
    assert sent == len(open_ids) + len(content_ids)


def test_send_merges_consecutive_turns_with_the_same_role():
    tok = _tok()
    fake = _FakeSession([])
    p = ChatMLConversation(tok, cast(Session, fake))

    asyncio.run(p.send("user", "hi"))
    fake.sent.clear()

    asyncio.run(p.send("user", " there"))

    # no turn_close()/turn_open() re-sent for the same role
    assert fake.sent == [tok.encode(" there").ids]


def test_send_closes_the_previous_turn_on_role_change():
    tok = _tok()
    fake = _FakeSession([])
    p = ChatMLConversation(tok, cast(Session, fake))

    asyncio.run(p.send("system", "sys"))
    fake.sent.clear()

    asyncio.run(p.send("user", "hi"))

    close_ids = tok.encode(p.turn_close()).ids
    open_ids = tok.encode(p.turn_open("user")).ids
    content_ids = tok.encode("hi").ids
    assert fake.sent == [close_ids, open_ids, content_ids]


def test_send_collapse_turn_skips_closing_the_previous_turn():
    tok = _tok()
    fake = _FakeSession([])
    p = ChatMLConversation(tok, cast(Session, fake))

    asyncio.run(p.send("assistant", ""))
    fake.sent.clear()

    asyncio.run(p.send("user", "hi", collapse_turn=True))

    open_ids = tok.encode(p.turn_open("user")).ids
    content_ids = tok.encode("hi").ids
    assert fake.sent == [open_ids, content_ids]
