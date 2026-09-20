"""Engine over KServe v2: chat template and usage are computed client-side."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ...client import AsyncTritonLLMClient
from ..convert import to_chat_completion, to_chat_messages, to_completion, to_sampling_params
from ..streaming import ChatStreamAssembler, CompletionStreamAssembler
from ..types import ChatCompletion, ChatCompletionChunk, Completion, Model, ModelList
from .base import include_usage


class NativeEngine:
    """Serves the openai-shaped API over KServe v2 via :class:`AsyncTritonLLMClient`."""

    def __init__(self, client: AsyncTritonLLMClient) -> None:
        """Bind the engine to a native Triton client.

        :param client: The client used to talk to Triton over HTTP or gRPC.
        """
        self.client = client

    async def chat(
        self, *, messages: list[Any], model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Render the conversation client-side and generate a completion for it.

        :param messages: Conversation messages, openai-style dicts or :class:`ChatMessage`.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments (sampling params, options).
        :return: A :class:`ChatCompletion` when ``stream`` is ``False``, otherwise an async
            iterator of :class:`ChatCompletionChunk`.
        """
        params = to_sampling_params(kwargs)
        prompt = self.client.render_chat(to_chat_messages(messages))
        resolved = self.client.resolve_model(model)
        template = self.client.chat_template
        if stream:
            assembler = ChatStreamAssembler(
                model=resolved,
                prompt=prompt,
                template=template,
                max_tokens=params.max_tokens,
                include_usage=include_usage(kwargs),
            )
            return self._chat_stream(prompt, params, resolved, assembler)
        response = await self.client.generate(prompt, params, model=resolved)
        return to_chat_completion(response, prompt=prompt, template=template, max_tokens=params.max_tokens)

    async def _chat_stream(
        self, prompt: str, params: Any, model: str, assembler: ChatStreamAssembler
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Drive the underlying stream and assemble it into ChatCompletionChunk objects.

        :param prompt: The rendered prompt text.
        :param params: Sampling parameters to use.
        :param model: Resolved model name.
        :param assembler: Assembler that turns SDK chunks into openai-shaped chunks.
        :return: An async iterator of :class:`ChatCompletionChunk`.
        """
        yield assembler.start()
        async for chunk in self.client.stream(prompt, params, model=model):
            piece = assembler.delta(chunk)
            if piece is not None:
                yield piece
        for piece in assembler.finish():
            yield piece

    async def completion(
        self, *, prompt: str, model: str | None, stream: bool, kwargs: dict[str, Any]
    ) -> Completion | AsyncIterator[Completion]:
        """Generate a legacy completion for a raw prompt.

        :param prompt: Raw prompt text.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Remaining openai-style keyword arguments (sampling params, options).
        :return: A :class:`Completion` when ``stream`` is ``False``, otherwise an async iterator
            of :class:`Completion` chunks.
        """
        params = to_sampling_params(kwargs)
        resolved = self.client.resolve_model(model)
        template = self.client.chat_template
        if stream:
            assembler = CompletionStreamAssembler(
                model=resolved,
                prompt=prompt,
                template=template,
                max_tokens=params.max_tokens,
                include_usage=include_usage(kwargs),
            )
            return self._completion_stream(prompt, params, resolved, assembler)
        response = await self.client.generate(prompt, params, model=resolved)
        return to_completion(response, prompt=prompt, template=template, max_tokens=params.max_tokens)

    async def _completion_stream(
        self, prompt: str, params: Any, model: str, assembler: CompletionStreamAssembler
    ) -> AsyncIterator[Completion]:
        """Drive the underlying stream and assemble it into Completion chunks.

        :param prompt: The rendered prompt text.
        :param params: Sampling parameters to use.
        :param model: Resolved model name.
        :param assembler: Assembler that turns SDK chunks into openai-shaped chunks.
        :return: An async iterator of :class:`Completion` chunks.
        """
        async for chunk in self.client.stream(prompt, params, model=model):
            piece = assembler.delta(chunk)
            if piece is not None:
                yield piece
        for piece in assembler.finish():
            yield piece

    async def models(self) -> ModelList:
        """List ready models from the Triton repository index.

        :return: The models listing, restricted to models in the ``READY`` state (or with an
            unknown state).
        """
        models = await self.client.list_models()
        return ModelList(data=[Model(id=item.name) for item in models if item.state in (None, "READY")])

    async def ready(self) -> bool:
        """Check overall server readiness.

        :return: ``True`` if the server is ready.
        """
        return await self.client.is_ready()

    async def close(self) -> None:
        """Close the underlying Triton client."""
        await self.client.close()
