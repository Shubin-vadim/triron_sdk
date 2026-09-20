"""Resource namespaces (``client.chat``, ``client.completions``, ``client.models``), async and sync."""

from .chat import AsyncChat, AsyncChatCompletions, Chat, ChatCompletions
from .completions import AsyncCompletions, Completions
from .models import AsyncModels, Models


__all__ = [
    "AsyncChat",
    "AsyncChatCompletions",
    "AsyncCompletions",
    "AsyncModels",
    "Chat",
    "ChatCompletions",
    "Completions",
    "Models",
]
