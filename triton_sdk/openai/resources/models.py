"""``client.models`` resource, async and sync."""

from __future__ import annotations

from ...core.sync_runner import AsyncRunner
from ..engines import Engine
from ..types import ModelList


class AsyncModels:
    """Async ``client.models`` resource."""

    def __init__(self, engine: Engine) -> None:
        """Bind the resource to an engine.

        :param engine: The engine that actually serves requests.
        """
        self._engine = engine

    async def list(self) -> ModelList:
        """List available models, mirroring ``openai.resources.Models.list``.

        :return: The models listing.
        """
        return await self._engine.models()


class Models:
    """Synchronous ``client.models`` resource."""

    def __init__(self, runner: AsyncRunner, inner: AsyncModels) -> None:
        """Bind the resource to a background runner and its async counterpart.

        :param runner: Runner used to drive the async resource from sync code.
        :param inner: The wrapped async resource.
        """
        self._runner = runner
        self._inner = inner

    def list(self) -> ModelList:
        """Synchronous equivalent of :meth:`AsyncModels.list`.

        :return: The models listing.
        """
        return self._runner.run(self._inner.list())
