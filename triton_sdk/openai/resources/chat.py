"""``client.chat.completions`` resource, async and sync."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, overload

from ...core.sync_runner import AsyncRunner
from ..engines import Engine
from ..types import ChatCompletion, ChatCompletionChunk


class AsyncChatCompletions:
    """Async ``client.chat.completions`` resource."""

    def __init__(self, engine: Engine) -> None:
        """Bind the resource to an engine.

        :param engine: The engine that actually serves requests.
        """
        self._engine = engine

    @overload
    async def create(
        self, *, messages: list[Any], model: str | None = None, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self, *, messages: list[Any], model: str | None = None, stream: Literal[True], **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def create(
        self, *, messages: list[Any], model: str | None = None, stream: bool = False, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Create a chat completion, mirroring ``openai.resources.chat.Completions.create``.

        :param messages: Conversation messages, openai-style dicts or :class:`ChatMessage`.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Sampling parameters and options (``max_tokens``, ``temperature``,
            ``stop``, ``stream_options``, ``extra_body``, ...).
        :return: A :class:`ChatCompletion` when ``stream`` is ``False``, otherwise an async
            iterator of :class:`ChatCompletionChunk`.
        :raises triton_sdk.openai.convert.UnsupportedParameterError: If an unsupported argument
            (e.g. ``tools``) is passed.
        :raises TypeError: If an unrecognized keyword argument is passed.
        """
        return await self._engine.chat(messages=messages, model=model, stream=stream, kwargs=kwargs)


class AsyncChat:
    """Async ``client.chat`` resource, holding :attr:`completions`."""

    def __init__(self, engine: Engine) -> None:
        """Bind the resource to an engine.

        :param engine: The engine that actually serves requests.
        """
        self.completions = AsyncChatCompletions(engine)


class ChatCompletions:
    """Synchronous ``client.chat.completions`` resource."""

    def __init__(self, runner: AsyncRunner, inner: AsyncChatCompletions) -> None:
        """Bind the resource to a background runner and its async counterpart.

        :param runner: Runner used to drive the async resource from sync code.
        :param inner: The wrapped async resource.
        """
        self._runner = runner
        self._inner = inner

    @overload
    def create(
        self, *, messages: list[Any], model: str | None = None, stream: Literal[False] = False, **kwargs: Any
    ) -> ChatCompletion: ...

    @overload
    def create(
        self, *, messages: list[Any], model: str | None = None, stream: Literal[True], **kwargs: Any
    ) -> Iterator[ChatCompletionChunk]: ...

    def create(
        self, *, messages: list[Any], model: str | None = None, stream: bool = False, **kwargs: Any
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Synchronous equivalent of :meth:`AsyncChatCompletions.create`.

        :param messages: Conversation messages, openai-style dicts or :class:`ChatMessage`.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Sampling parameters and options.
        :return: A :class:`ChatCompletion` when ``stream`` is ``False``, otherwise a
            synchronous iterator of :class:`ChatCompletionChunk`.
        """
        result = self._runner.run(self._inner.create(messages=messages, model=model, stream=stream, **kwargs))  # type: ignore[call-overload]
        if stream:
            return self._runner.iterate(result)
        return result  # type: ignore[no-any-return]


class Chat:
    """Synchronous ``client.chat`` resource, holding :attr:`completions`."""

    def __init__(self, runner: AsyncRunner, inner: AsyncChat) -> None:
        """Bind the resource to a background runner and its async counterpart.

        :param runner: Runner used to drive the async resource from sync code.
        :param inner: The wrapped async resource.
        """
        self.completions = ChatCompletions(runner, inner.completions)
