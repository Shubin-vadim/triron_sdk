"""Runs an asyncio event loop in a daemon thread so async clients can be driven from synchronous code."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Awaitable, Iterator
from typing import TypeVar


T = TypeVar("T")


class AsyncRunner:
    """Owns a background asyncio event loop and lets sync callers drive coroutines on it."""

    def __init__(self, name: str = "triton-sdk-loop") -> None:
        """Start the background event loop thread.

        :param name: Thread name, useful for debugging with multiple runners alive at once.
        """
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()

    def run(self, coro: Awaitable[T]) -> T:
        """Run a coroutine on the background loop and block until it completes.

        :param coro: The coroutine (or other awaitable) to run.
        :return: The awaitable's result.
        :raises Exception: Whatever exception the awaitable raises, re-raised in the calling thread.
        """
        return asyncio.run_coroutine_threadsafe(_await(coro), self._loop).result()

    def iterate(self, agen: AsyncGenerator[T, None]) -> Iterator[T]:
        """Drive an async generator step by step from synchronous code.

        Closing the returned iterator early (e.g. via ``break``) closes the underlying async
        generator, which propagates cancellation to whatever it was awaiting.

        :param agen: The async generator to drive.
        :return: A synchronous iterator yielding the same values.
        """
        try:
            while True:
                yield self.run(anext(agen))
        except StopAsyncIteration:
            return
        finally:
            self.run(agen.aclose())

    def close(self) -> None:
        """Stop the background loop and join its thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


async def _await[T](coro: Awaitable[T]) -> T:
    """Await a coroutine; a plain indirection point so it can be scheduled with ``run_coroutine_threadsafe``.

    :param coro: The awaitable to await.
    :return: Its result.
    """
    return await coro
