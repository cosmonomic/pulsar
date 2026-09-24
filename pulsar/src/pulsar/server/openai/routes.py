# TODO: refactor. the openai api endpoint should
# map conversation history to a session_id,
# see pulsar.conversation
#
# we naively don't fork / cache prefixes.
# we keep a tree of history digests
# where digest(message_t, message_{t-1}..message_0) = combine(message_t, digest(message_{t-2}..message_0))
# for a history of messages t=0..N.
# Note that history can have the same prefix messages.
# We store a mapping digest to list[session_id].
# For requests, we compute the digest of messages t-1 to 0
# and check in the mapping if there is a session there;
# if yes, continue the session.
# if no, walk up the tree until we find a conversation, then fork it (and adding to the list of session_ids at that digest mapping entry)
# Meaning that sessions are continued only if its incremented with a single turn,
# otherwise it is forked and prefilled until up to date.
#
# This also means that user's agent might be using a session A for some turns,
# and due to forking, ends up using session B at a later time. But this is fine,
# since forking should make it so that A and B are indistinguishable.
# Thus session is not shown to the user.
# The native API users will get token that maps to the most recent digest of the conversation,
# which is used to retrieve a session that can fulfill the user's request.
import functools
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal, Annotated

from fastapi import HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter

from pulsar.conversation import (
    ChatMLConversation,
    ConversationStore,
    StoredConversation,
    TruncatedError,
)
from pulsar.server.dependencies import Pulsar
from pulsar.server.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    Usage,
)


@dataclass
class ServerState:
    pulsar: Pulsar
    model_id: str
    max_sessions: int
    conversations: ConversationStore = field(init=False)

    def __post_init__(self) -> None:
        self.conversations = ConversationStore(
            self.model_id,
            self.max_sessions,
            functools.partial(
                ChatMLConversation.from_new_session,
                self.pulsar.tokenizer,
                self.pulsar.create_session,
            ),
        )

def get_server_state(request: Request) -> ServerState:
    return request.app.state.server_state

type State = Annotated[ServerState, Depends(get_server_state)]


router = APIRouter()


@router.get("/models")
async def list_models(state: State) -> ModelList:
    return ModelList(data=[ModelCard(id=state.model_id, created=int(time.time()))])


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, state: State):
    if body.model != state.model_id:
        raise HTTPException(404, f"model {body.model!r} not found, only {state.model_id} is loaded")
    if not body.messages:
        raise HTTPException(422, "messages must be non-empty")

    conversation, prompt_tokens = await _prefill(state, body.messages)
    max_tokens = sys.maxsize if body.max_tokens is None else body.max_tokens

    if body.stream:
        return await _stream_completion(body, conversation, max_tokens)

    return await _full_completion(body, conversation, max_tokens, prompt_tokens)


async def _prefill(
    state: ServerState, messages: list[ChatMessage]
) -> tuple[StoredConversation, int]:
    *prefix, last = messages

    with state.conversations.claim_by_history(
        (m.role, m.content) for m in prefix
    ) as claimed:
        if claimed is not None:
            conversation = claimed
            prompt_tokens = 0
            rest = [last]
            for m in rest:
                prompt_tokens += await conversation.send(m.role, m.content)
            return conversation, prompt_tokens
        else:
            first, *rest = messages
            if first.role == "system":
                system_prompt = first.content
            else:
                system_prompt = ""
                rest = messages
            async with state.conversations.create(system_prompt) as (
                prompt_tokens,
                conversation,
            ):
                for m in rest:
                    prompt_tokens += await conversation.send(m.role, m.content)
                return conversation, prompt_tokens


async def _full_completion(
    body: ChatCompletionRequest,
    conversation: StoredConversation,
    max_tokens: int,
    prompt_tokens: int,
) -> ChatCompletionResponse:
    parts: list[str] = []
    finish_reason: Literal["stop", "length"] = "stop"
    try:
        async for text in conversation.recv(max_tokens):
            parts.append(text)
    except TruncatedError:
        finish_reason = "length"

    completion_tokens = len(parts)
    reply = ChatMessage(role="assistant", content="".join(parts))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatCompletionChoice(index=0, message=reply, finish_reason=finish_reason)
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_completion(
    body: ChatCompletionRequest,
    conversation: StoredConversation,
    max_tokens: int,
) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(
        delta: ChatCompletionChunkDelta,
        finish_reason: Literal["stop", "length"] | None = None,
    ) -> str:
        payload = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=body.model,
            choices=[
                ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)
            ],
        )
        return f"data: {payload.model_dump_json()}\n\n"

    async def events() -> AsyncIterator[str]:
        yield chunk(ChatCompletionChunkDelta(role="assistant"))

        finish_reason: Literal["stop", "length"] = "stop"
        try:
            async for text in conversation.recv(max_tokens):
                if text:
                    yield chunk(ChatCompletionChunkDelta(content=text))
        except TruncatedError:
            finish_reason = "length"

        yield chunk(ChatCompletionChunkDelta(), finish_reason)
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
