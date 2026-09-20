"""``client.completions`` resource (legacy, non-chat completions), async and sync."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal, overload

from ...core.sync_runner import AsyncRunner
from ..engines import Engine
from ..types import Completion


class AsyncCompletions:
    """Async ``client.completions`` resource (legacy, non-chat completions)."""

    def __init__(self, engine: Engine) -> None:
        """Bind the resource to an engine.

        :param engine: The engine that actually serves requests.
        """
        self._engine = engine

    @overload
    async def create(
        self, *, prompt: str, model: str | None = None, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    async def create(
        self, *, prompt: str, model: str | None = None, stream: Literal[True], **kwargs: Any
    ) -> AsyncIterator[Completion]: ...

    async def create(
        self, *, prompt: str, model: str | None = None, stream: bool = False, **kwargs: Any
    ) -> Completion | AsyncIterator[Completion]:
        """Create a legacy completion, mirroring ``openai.resources.Completions.create``.

        :param prompt: Raw prompt text.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Sampling parameters and options.
        :return: A :class:`Completion` when ``stream`` is ``False``, otherwise an async iterator
            of :class:`Completion` chunks.
        """
        return await self._engine.completion(prompt=prompt, model=model, stream=stream, kwargs=kwargs)


class Completions:
    """Synchronous ``client.completions`` resource (legacy, non-chat completions)."""

    def __init__(self, runner: AsyncRunner, inner: AsyncCompletions) -> None:
        """Bind the resource to a background runner and its async counterpart.

        :param runner: Runner used to drive the async resource from sync code.
        :param inner: The wrapped async resource.
        """
        self._runner = runner
        self._inner = inner

    @overload
    def create(
        self, *, prompt: str, model: str | None = None, stream: Literal[False] = False, **kwargs: Any
    ) -> Completion: ...

    @overload
    def create(
        self, *, prompt: str, model: str | None = None, stream: Literal[True], **kwargs: Any
    ) -> Iterator[Completion]: ...

    def create(
        self, *, prompt: str, model: str | None = None, stream: bool = False, **kwargs: Any
    ) -> Completion | Iterator[Completion]:
        """Synchronous equivalent of :meth:`AsyncCompletions.create`.

        :param prompt: Raw prompt text.
        :param model: Model name; the client's configured default when omitted.
        :param stream: Whether to stream the response.
        :param kwargs: Sampling parameters and options.
        :return: A :class:`Completion` when ``stream`` is ``False``, otherwise a synchronous
            iterator of :class:`Completion` chunks.
        """
        result = self._runner.run(self._inner.create(prompt=prompt, model=model, stream=stream, **kwargs))  # type: ignore[call-overload]
        if stream:
            return self._runner.iterate(result)
        return result  # type: ignore[no-any-return]
