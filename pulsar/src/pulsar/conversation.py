from __future__ import annotations

import hashlib
import sys
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from contextlib import asynccontextmanager, contextmanager
from functools import cache
from typing import Literal, Self, final

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from pulsar.rt.session import Session


class TruncatedError(Exception):
    """Raised by Conversation.recv() once max_tokens or the decode limit is reached."""


class Conversation(ABC):
    type Role = Literal["system", "user", "assistant"]
    type TurnState = Role | None

    def __init__(
        self,
        tokenizer: Tokenizer,
        session: Session,
        turn: "Conversation.TurnState" = None,
    ):
        self._tokenizer = tokenizer
        self._session = session
        self._turn: Conversation.TurnState = turn

    @classmethod
    async def from_new_session(
        cls,
        tokenizer: Tokenizer,
        session_factory: Callable[[list[int]], Awaitable[Session]],
        system_prompt: str,
    ) -> tuple[int, Self]:
        turn = [cls.turn_open("system"), system_prompt, cls.turn_close()]
        tokens = tokenizer.encode("".join(turn)).ids
        session = await session_factory(tokens)
        return len(tokens), cls(tokenizer, session)

    @property
    def session(self) -> Session:
        return self._session

    @property
    @abstractmethod
    def stop_token_ids(self) -> list[int]: ...

    @classmethod
    @abstractmethod
    def turn_open(cls, role: Role) -> str: ...

    @classmethod
    @abstractmethod
    def turn_close(cls) -> str: ...

    async def _feed(self, message: str) -> int:
        tokens = self._tokenizer.encode(message).ids
        await self._session.send(tokens)
        return len(tokens)

    async def send(
        self,
        role: Role,
        content: str,
        collapse_turn: bool = False,
    ) -> int:
        tokens = 0
        if role != self._turn:
            if self._turn is not None and not collapse_turn:
                tokens += await self._feed(self.turn_close())
            tokens += await self._feed(self.turn_open(role))
            self._turn = role
        tokens += await self._feed(content)
        return tokens

    async def recv(self, max_tokens: int = sys.maxsize) -> AsyncGenerator[str]:
        # TODO implement max_tokens in engine internals
        await self.send("assistant", "")
        stream = await self._session.recv()

        decoder = DecodeStream(skip_special_tokens=False)
        async for token_id in stream:
            if token_id in self.stop_token_ids:
                # The model's own stop token already closes the turn in the
                # session's KV state, so no synthetic turn_close() is needed.
                self._turn = None
                yield ""
                return
            yield decoder.step(self._tokenizer, token_id) or ""

        raise TruncatedError("decode limit reached")


@final
class ChatMLConversation(Conversation):
    turn_open_marker_format = "<|im_start|>{role}\n"
    turn_close_markers = ["<|im_end|>", "<|endoftext|>"]

    @property
    @cache
    def stop_token_ids(self) -> list[int]:
        def token_to_id(token: str):
            token_id = self._tokenizer.token_to_id(token)
            assert token_id is not None
            return token_id

        return [token_to_id(x) for x in self.turn_close_markers]

    @classmethod
    def turn_open(cls, role: str) -> str:
        return cls.turn_open_marker_format.format(role=role)

    @classmethod
    def turn_close(cls) -> str:
        return "<|im_end|>\n"


type ConversationFactory = Callable[[str], Awaitable[tuple[int, Conversation]]]


class HistoryDigest:
    """Chains a per-conversation digest of (role, content) turns and indexes
    conversations by that digest, so a caller holding only a message history
    (no conversation_id) can resolve the conversation it belongs to.

    update() advances a conversation's digest by exactly the one new turn given,
    chaining onto whatever digest is already on file for it (or the model's
    genesis digest, the first time a conversation_id is updated). A caller with n
    new turns to record (e.g. a new user message and the reply it produced)
    calls update() once per turn; nothing needs to re-hash the whole history
    the way resolve() must, since callers don't retain a digest between
    separate requests but this object does between calls.

    seed() sets a conversation_id's starting digest without making it
    resolvable by that digest alone: it writes _digests (conversation_id ->
    digest) but not _conversation_ids (digest -> conversation_id), the
    reverse of what update() keeps in sync. That matters for a conversation
    whose first turn is always the same content across every conversation
    with the same genesis (e.g. an empty system prompt): were that digest
    resolvable on its own, every such conversation would share it the
    instant it's seeded, before any turn distinguishes it from another one
    seeded the same way, letting a concurrent, unrelated conversation's
    history lookup match it by pure timing. update()'s first call for that
    conversation_id chains onto the seeded digest exactly as it would onto
    any other, so it's resolvable from then on like normal, once combined
    with a real turn.
    """

    def __init__(self, genesis_key: str) -> None:
        self._genesis = hashlib.sha256(genesis_key.encode()).digest()
        self._conversation_ids: dict[str, str] = {}  # digest -> conversation_id
        self._digests: dict[str, bytes] = {}  # conversation_id -> digest, the reverse

    @staticmethod
    def _step(digest: bytes, role: Conversation.Role, content: str) -> bytes:
        h = hashlib.sha256(digest)
        for field in (role.encode(), content.encode()):
            h.update(len(field).to_bytes(8, "big"))
            h.update(field)
        return h.digest()

    def _digest_of(self, history: Iterable[tuple[Conversation.Role, str]]) -> str:
        digest = self._genesis
        for role, content in history:
            digest = self._step(digest, role, content)
        return digest.hex()

    def resolve(self, history: Iterable[tuple[Conversation.Role, str]]) -> str | None:
        return self._conversation_ids.get(self._digest_of(history))

    def seed(self, conversation_id: str, turn: tuple[Conversation.Role, str]) -> None:
        self._digests[conversation_id] = self._step(self._genesis, *turn)

    def update(self, conversation_id: str, turn: tuple[Conversation.Role, str]) -> None:
        previous = self._digests.get(conversation_id)
        digest = self._step(previous if previous is not None else self._genesis, *turn)
        if previous is not None:
            self._conversation_ids.pop(previous.hex(), None)
        self._conversation_ids[digest.hex()] = conversation_id
        self._digests[conversation_id] = digest


class ConversationStore:
    """Caches active Conversations by an opaque conversation_id.

    Every way of getting a StoredConversation out of this store -- claim(),
    claim_by_history(), create() -- is a context manager, not a plain
    accessor: holding a StoredConversation without a corresponding release
    is exactly what "claimed" means here, so the only way to obtain one is
    to also commit, syntactically, to releasing it. There's no separate
    close()/release() to forget to call.

    claim() is a pure conversation_id lookup, for a caller that already
    tracks its own conversation_id; it's still a context manager because
    holding the id doesn't mean holding the claim. claim_by_history() layers
    history.resolve() on top of it, for a caller (e.g. an API wrapper) that
    only has a message history. Both yield None for a conversation_id that's
    currently claimed by someone else, the same as for one that doesn't
    exist: a StoredConversation isn't safe for two concurrent callers to
    send()/recv() on at once (it mutates shared KV session state with no
    locking of its own), so a caller that loses the race for a busy
    conversation_id is treated as not having found it, e.g. free to create()
    a new one instead of corrupting the conversation an in-flight request is
    still using.

    create() is the only way to get a new StoredConversation: it opens the
    underlying Conversation via conversation_factory, mints the
    conversation_id, and registers it, so a caller can't end up with a
    StoredConversation the store doesn't know about. It also claims the
    conversation_id on the caller's behalf: whoever creates a conversation
    already holds it, the same as whoever wins a claim()/claim_by_history()
    lookup.

    create() seeds the opening system turn via history.seed() rather than
    history.update(): see HistoryDigest for why. Its first real turn, sent
    through the ordinary StoredConversation.send()/recv() path, is what
    makes the conversation resolvable by history.

    claim_by_history() mirrors that: a history that doesn't already start
    with a system turn gets one synthesized ("system", "") onto the front
    before resolving, so a client that never sent a system message still
    matches the conversation create() opened for it.

    conversation_id is a fresh uuid minted here, not the engine's internal seq_id,
    which is engine-local and won't survive a future export/import or
    sharding scheme.

    TODO: claim_by_history() grants continuation on content match alone,
    with no proof the caller sending this history is the one who produced
    it. An unrelated caller whose message history happens to match an
    existing, currently-unclaimed conversation's digest claims it and
    mutates its live KV state; the rightful continuer's next request then
    computes a digest that's moved on, misses, and silently forks into an
    unrelated new conversation. The busy check in claim()/claim_by_history()
    only rules out two claimants overlapping in time; it does nothing for
    this sequential case. Fixing it needs either replaying the whole
    matched history into a fresh session on every claim_by_history() hit
    (correct, but a full re-prefill every time, defeating the point of
    matching by history instead of just replaying the client's own message
    list), or giving the client a conversation_id to claim by id instead of
    by content -- there's no session fork/copy-on-write primitive in the
    engine today to make the replay path cheap.
    """

    def __init__(
        self, model_id: str, capacity: int, conversation_factory: ConversationFactory
    ) -> None:
        # capacity doesn't have any effect right now
        # when implemented, it should move older conversations to disk
        self._capacity = capacity
        self._conversation_factory = conversation_factory
        self._conversations: OrderedDict[str, StoredConversation] = OrderedDict()
        self._claimed: set[str] = set()
        self.history = HistoryDigest(model_id)

    @staticmethod
    def new_conversation_id() -> str:
        return str(uuid.uuid4())

    @contextmanager
    def claim(self, conversation_id: str) -> Iterator[StoredConversation | None]:
        if conversation_id in self._claimed:
            yield None
            return
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            yield None
            return
        self._claimed.add(conversation_id)
        try:
            yield conversation
        finally:
            self._claimed.discard(conversation_id)

    @contextmanager
    def claim_by_history(
        self, history: Iterable[tuple[Conversation.Role, str]]
    ) -> Iterator[StoredConversation | None]:
        turns = list(history)
        if not turns or turns[0][0] != "system":
            turns.insert(0, ("system", ""))
        conversation_id = self.history.resolve(turns)
        if conversation_id is None:
            yield None
            return
        with self.claim(conversation_id) as conversation:
            yield conversation

    @asynccontextmanager
    async def create(
        self, system_prompt: str
    ) -> AsyncIterator[tuple[int, StoredConversation]]:
        prompt_tokens, conversation = await self._conversation_factory(system_prompt)
        conversation_id = self.new_conversation_id()
        self.history.seed(conversation_id, ("system", system_prompt))
        wrapped = StoredConversation(
            self.history,
            conversation_id,
            conversation,
            release=lambda: self._claimed.discard(conversation_id),
        )
        self._conversations[conversation_id] = wrapped
        self._claimed.add(conversation_id)
        try:
            yield prompt_tokens, wrapped
        finally:
            self._claimed.discard(conversation_id)


@final
class StoredConversation:
    """A Conversation bound to a conversation_id and a HistoryDigest.

    Wraps (not subclasses) a Conversation so that every client-facing turn
    is recorded via HistoryDigest.update() at the point it's sent or
    received, instead of requiring a caller to remember a matching update()
    call alongside every send()/recv(). Composition, not inheritance,
    matters here: Conversation.recv() opens the assistant turn internally
    with self.send("assistant", ""), and that internal call must not be
    recorded (it's session plumbing, not a client-visible turn). Subclassing
    Conversation would route that internal call through the override too;
    holding the inner Conversation instead keeps it out of reach.

    Holds only the HistoryDigest, not the whole ConversationStore: it has no
    business reaching into the store's conversation_id -> Conversation cache, and
    the store already holds a reference to each StoredConversation it hands
    out, so holding the store back would be a cycle for no benefit.

    Constructed only by ConversationStore.create(); the store is what's
    responsible for minting the conversation_id, seeding its starting
    digest, and registering the result.

    The claim it's held under is normally released by exiting whichever of
    ConversationStore's claim()/claim_by_history()/create() context
    managers produced it. close() is an early-release escape hatch for a
    caller done with it before that block ends, the same as a file object
    supports both `with open(...) as f` and a direct f.close(). Closing
    more than once is a no-op. Closing early and then continuing to use
    the enclosing block as though it still exclusively holds the claim is
    on the caller: nothing stops a new claimant from grabbing conversation_id
    the moment it's released, early or not.
    """

    def __init__(
        self,
        history: HistoryDigest,
        conversation_id: str,
        conversation: Conversation,
        release: Callable[[], None],
    ) -> None:
        self._history = history
        self._conversation_id = conversation_id
        self._conversation = conversation
        self._release = release

    @property
    def conversation_id(self) -> str:
        return self._conversation_id

    def close(self) -> None:
        self._release()

    async def send(
        self,
        role: Conversation.Role,
        content: str,
        collapse_turn: bool = False,
    ) -> int:
        tokens = await self._conversation.send(role, content, collapse_turn)
        self._history.update(self._conversation_id, (role, content))
        return tokens

    async def recv(self, max_tokens: int = sys.maxsize) -> AsyncGenerator[str]:
        parts: list[str] = []
        try:
            async for text in self._conversation.recv(max_tokens):
                parts.append(text)
                yield text
        finally:
            self._history.update(self._conversation_id, ("assistant", "".join(parts)))
