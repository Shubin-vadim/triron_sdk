"""Response types shaped like the openai SDK (openai.types.chat.*, openai.types.completion.*).

Only the fields Triton can populate are modelled; attribute names and nesting match openai so code written
against `response.choices[0].message.content` / `chunk.choices[0].delta.content` ports unchanged.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FinishReason = Literal["stop", "length", "content_filter", "tool_calls", "function_call"]


def new_id(prefix: str) -> str:
    """Generate an openai-style random id, e.g. ``chatcmpl-3f9a...``.

    :param prefix: Id prefix (e.g. ``"chatcmpl"`` or ``"cmpl"``).
    :return: The prefix, a hyphen, and 24 hex characters.
    """
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now() -> int:
    """Current Unix timestamp, in seconds.

    :return: The current time as an integer Unix timestamp.
    """
    return int(time.time())


class CompletionUsage(BaseModel):
    """Prompt/completion token counts, mirroring ``openai.types.CompletionUsage``."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionMessage(BaseModel):
    """One assistant message, mirroring ``openai.types.chat.ChatCompletionMessage``."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class Choice(BaseModel):
    """One completion choice of a non-streaming chat response."""

    index: int = 0
    message: ChatCompletionMessage
    finish_reason: FinishReason | None = None
    logprobs: object | None = None


class ChatCompletion(BaseModel):
    """Non-streaming chat completion, mirroring ``openai.types.chat.ChatCompletion``."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=now)
    model: str
    choices: list[Choice]
    usage: CompletionUsage | None = None


class ChoiceDelta(BaseModel):
    """The incremental content of one streamed chat chunk."""

    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """One completion choice of a streaming chat chunk."""

    index: int = 0
    delta: ChoiceDelta
    finish_reason: FinishReason | None = None
    logprobs: object | None = None


class ChatCompletionChunk(BaseModel):
    """One chunk of a streamed chat completion, mirroring ``openai.types.chat.ChatCompletionChunk``."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: CompletionUsage | None = None


class CompletionChoice(BaseModel):
    """One completion choice of a legacy (non-chat) completion."""

    index: int = 0
    text: str
    finish_reason: FinishReason | None = None
    logprobs: object | None = None


class Completion(BaseModel):
    """Legacy /v1/completions shape. Streaming reuses it with partial `text` per chunk, like openai does."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: new_id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=now)
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage | None = None


class Model(BaseModel):
    """One entry of a models listing, mirroring ``openai.types.Model``."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=now)
    owned_by: str = "triton"


class ModelList(BaseModel):
    """A models listing, mirroring ``openai.pagination.SyncPage[Model]``."""

    object: Literal["list"] = "list"
    data: list[Model]
