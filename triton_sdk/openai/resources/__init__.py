"""Resource namespaces (``client.chat``, ``client.completions``, ``client.models``), async and sync."""

from triton_sdk.openai.resources.chat import AsyncChat, AsyncChatCompletions, Chat, ChatCompletions
from triton_sdk.openai.resources.completions import AsyncCompletions, Completions
from triton_sdk.openai.resources.models import AsyncModels, Models


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
