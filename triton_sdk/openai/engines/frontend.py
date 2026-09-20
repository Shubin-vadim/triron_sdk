"""Engine over Triton's OpenAI frontend: requests are forwarded almost verbatim."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..frontend import AsyncOpenAIFrontend
from ..types import ChatCompletion, ChatCompletionChunk, Completion, ModelList


class FrontendEngine:
    """Serves the openai-shaped API by forwarding requests to Triton's OpenAI frontend."""

    def __init__(self, frontend: AsyncOpenAIFrontend, *, default_model: str | None = None) -> None:
        """Bind the engine to a frontend client.

        :param frontend: The client used to talk to Triton's OpenAI frontend.
        :param default_model: Model name to use when a call omits ``model``.
        """
        self.frontend = frontend
        self._default_model = default_model

    def _body(self, model: str | None, kwargs: dict[str, Any], **fields: Any) -> dict[str, Any]:
        """Build the JSON body to forward to the frontend.

        :param model: Model name passed by the caller, or ``None`` to use :attr:`_default_model`.
        :param kwargs: openai-style keyword arguments; ``extra_body`` is merged in and
            ``timeout``/``extra_headers`` are dropped (they are not part of the wire request).
        :param fields: Extra top-level fields to set (e.g. ``messages`` or ``prompt``).
        :return: The request body to send to the frontend.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        """
        resolved = model or self._default_model
        if not resolved:
            msg = "model is required: pass model=... or set TRITON_DEFAULT_MODEL"
            raise ValueError(msg)
        body = {key: value for key, value in kwargs.items() if key not in {"extra_body", "timeout", "extra_headers"}}
        body.update(kwargs.get("extra_body") or {})
        body.update(fields, model=resolved)
        return body

    async def chat(
        self, *, messages: list[Any], model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Forward a chat completion request to the frontend almost verbatim.

        :param messages: Conversation messages, openai-style dicts or :class:`ChatMessage`.
        :param model: Model name; :attr:`_default_model` when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments, forwarded as-is.
        :return: A :class:`ChatCompletion` when ``stream`` is ``False``, otherwise an async
            iterator of :class:`ChatCompletionChunk`.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        """
        payload_messages = [
            message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else message for message in messages
        ]
        body = self._body(model, kwargs, messages=payload_messages)
        if stream:
            return self._chat_stream(body)
        return ChatCompletion.model_validate(await self.frontend.chat_completions(body))

    async def _chat_stream(self, body: dict[str, Any]) -> AsyncIterator[ChatCompletionChunk]:
        """Stream the frontend's chat completion response, validated into our types.

        :param body: Request body to send.
        :return: An async iterator of :class:`ChatCompletionChunk`.
        """
        async for event in self.frontend.chat_completions_stream(body):
            yield ChatCompletionChunk.model_validate(event)

    async def completion(
        self, *, prompt: str, model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> Completion | AsyncIterator[Completion]:
        """Forward a legacy completion request to the frontend almost verbatim.

        :param prompt: Raw prompt text.
        :param model: Model name; :attr:`_default_model` when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments, forwarded as-is.
        :return: A :class:`Completion` when ``stream`` is ``False``, otherwise an async iterator
            of :class:`Completion` chunks.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        """
        body = self._body(model, kwargs, prompt=prompt)
        if stream:
            return self._completion_stream(body)
        return Completion.model_validate(await self.frontend.completions(body))

    async def _completion_stream(self, body: dict[str, Any]) -> AsyncIterator[Completion]:
        """Stream the frontend's completion response, validated into our types.

        :param body: Request body to send.
        :return: An async iterator of :class:`Completion` chunks.
        """
        async for event in self.frontend.completions_stream(body):
            yield Completion.model_validate(event)

    async def models(self) -> ModelList:
        """List models as reported by the frontend.

        :return: The models listing.
        """
        return ModelList.model_validate(await self.frontend.models())

    async def ready(self) -> bool:
        """Check frontend readiness.

        :return: ``True`` if the frontend is ready.
        """
        return await self.frontend.ready()

    async def close(self) -> None:
        """Close the underlying frontend client."""
        await self.frontend.close()
