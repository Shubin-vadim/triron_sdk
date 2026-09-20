"""Engine contract: the two ways of serving the openai-shaped API share this interface.

NativeEngine   - KServe v2 (HTTP/gRPC) through AsyncTritonLLMClient; chat template and usage computed client-side.
FrontendEngine - Triton's OpenAI frontend (/v1/chat/completions); the request is forwarded almost verbatim and the
                 server does templating and token accounting.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from ..types import ChatCompletion, ChatCompletionChunk, Completion, ModelList


class Engine(Protocol):
    """Common interface implemented by NativeEngine and FrontendEngine."""

    async def chat(
        self, *, messages: list[Any], model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Serve a chat completion request.

        :param messages: Conversation messages, openai-style dicts or :class:`ChatMessage`.
        :param model: Model name; engine-specific default applies when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments (sampling params, options).
        :return: A :class:`ChatCompletion` when ``stream`` is ``False``, otherwise an async
            iterator of :class:`ChatCompletionChunk`.
        """
        ...

    async def completion(
        self, *, prompt: str, model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> Completion | AsyncIterator[Completion]:
        """Serve a legacy completion request.

        :param prompt: Raw prompt text.
        :param model: Model name; engine-specific default applies when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments (sampling params, options).
        :return: A :class:`Completion` when ``stream`` is ``False``, otherwise an async iterator
            of :class:`Completion` chunks.
        """
        ...

    async def models(self) -> ModelList:
        """List available models.

        :return: The models listing.
        """
        ...

    async def ready(self) -> bool:
        """Check server/frontend readiness.

        :return: ``True`` if ready to serve requests.
        """
        ...

    async def close(self) -> None:
        """Release any underlying connections."""
        ...


def include_usage(kwargs: dict[str, Any]) -> bool:
    """Check whether the caller asked for a trailing usage chunk in a streamed response.

    :param kwargs: openai-style keyword arguments, possibly containing ``stream_options``.
    :return: ``True`` if ``stream_options.include_usage`` was set truthy.
    """
    return bool((kwargs.get("stream_options") or {}).get("include_usage"))
