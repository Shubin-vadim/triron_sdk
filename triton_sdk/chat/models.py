from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Role(StrEnum):
    """A chat message's author role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """One message in a chat conversation, before it is rendered into a raw prompt."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: str | None = None
