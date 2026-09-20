"""Synchronous facade over AsyncOpenAICompatClient. Runs its own event loop in a background thread."""

from __future__ import annotations

from typing import Any, Self

from ...client import TritonLLMClient
from ...core.sync_runner import AsyncRunner
from ..resources import Chat, Completions, Models
from .async_client import AsyncOpenAICompatClient


class OpenAICompatClient:
    """Synchronous facade over :class:`AsyncOpenAICompatClient`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start a background event loop and build the wrapped async client on it.

        :param args: Positional arguments forwarded to :class:`AsyncOpenAICompatClient`.
        :param kwargs: Keyword arguments forwarded to :class:`AsyncOpenAICompatClient`.
        """
        self._runner = AsyncRunner()
        # Constructed inside the loop so httpx / grpc.aio bind to the right event loop.
        self._async: AsyncOpenAICompatClient = self._runner.run(self._create(*args, **kwargs))
        self.chat = Chat(self._runner, self._async.chat)
        self.completions = Completions(self._runner, self._async.completions)
        self.models = Models(self._runner, self._async.models)

    @staticmethod
    async def _create(*args: Any, **kwargs: Any) -> AsyncOpenAICompatClient:
        """Build the wrapped async client; run on the background loop by the constructor.

        :param args: Positional arguments forwarded to :class:`AsyncOpenAICompatClient`.
        :param kwargs: Keyword arguments forwarded to :class:`AsyncOpenAICompatClient`.
        :return: The constructed async client.
        """
        return AsyncOpenAICompatClient(*args, **kwargs)

    @property
    def native(self) -> TritonLLMClient | None:
        """The underlying Triton client, wrapped for synchronous use.

        :return: A :class:`~triton_sdk.client.sync_client.TritonLLMClient` facade sharing this
            client's background loop, or ``None`` when talking to the OpenAI frontend.
        """
        if self._async.native is None:
            return None
        return TritonLLMClient.from_async(self._async.native, self._runner)

    def is_ready(self) -> bool:
        """Check server/frontend readiness.

        :return: ``True`` if ready to serve requests.
        """
        return self._runner.run(self._async.is_ready())

    def __enter__(self) -> Self:
        """Enter the context manager.

        :return: This client instance.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, closing the client.

        :param exc_info: Standard ``(exc_type, exc_value, traceback)`` tuple, unused.
        """
        self.close()

    def close(self) -> None:
        """Close the wrapped async client and stop the background loop."""
        self._runner.run(self._async.close())
        self._runner.close()
