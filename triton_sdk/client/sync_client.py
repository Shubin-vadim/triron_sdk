"""Sync facade over AsyncTritonLLMClient. Runs its own event loop in a background thread."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

from ..chat import ChatMessage, ChatTemplate
from ..core.models import GenerateChunk, GenerateResponse, ModelConfig, ModelMetadata, RepositoryModel, SamplingParams
from ..core.sync_runner import AsyncRunner
from .async_client import AsyncTritonLLMClient


class TritonLLMClient:
    """Synchronous facade over :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Start a background event loop and build the wrapped async client on it.

        :param args: Positional arguments forwarded to
            :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`.
        :param kwargs: Keyword arguments forwarded to
            :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`.
        """
        self._runner = AsyncRunner()
        self._owns_runner = True
        # Constructed inside the loop so httpx / grpc.aio bind to the right event loop.
        self._async: AsyncTritonLLMClient = self._runner.run(self._create(*args, **kwargs))

    @staticmethod
    async def _create(*args: Any, **kwargs: Any) -> AsyncTritonLLMClient:
        """Build the wrapped async client; run on the background loop by the constructor.

        :param args: Positional arguments forwarded to
            :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`.
        :param kwargs: Keyword arguments forwarded to
            :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`.
        :return: The constructed async client.
        """
        return AsyncTritonLLMClient(*args, **kwargs)

    @classmethod
    def from_async(cls, client: AsyncTritonLLMClient, runner: AsyncRunner) -> TritonLLMClient:
        """Wrap an existing async client that already lives on ``runner``'s loop.

        The caller keeps owning both ``client`` and ``runner``; the returned facade will not
        close them.

        :param client: The async client to wrap.
        :param runner: The runner whose loop ``client`` was created on.
        :return: A :class:`TritonLLMClient` facade over ``client``.
        """
        instance = cls.__new__(cls)
        instance._runner = runner  # noqa: SLF001 - alternate constructor
        instance._owns_runner = False  # noqa: SLF001
        instance._async = client  # noqa: SLF001
        return instance

    # --- plumbing ----------------------------------------------------------
    def _run(self, coro: Any) -> Any:
        """Run a coroutine on the background loop and block until it completes.

        :param coro: The coroutine to run.
        :return: The coroutine's result.
        """
        return self._runner.run(coro)

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
        """Close the wrapped async client and stop the background loop.

        A no-op when this facade was built via :meth:`from_async`, since it does not own the
        client or the runner in that case.
        """
        if not self._owns_runner:
            return
        self._run(self._async.close())
        self._runner.close()

    # --- mirrored API ------------------------------------------------------
    def is_ready(self, model: str | None = None) -> bool:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.is_ready`.

        :param model: Model name to check; when omitted, checks overall server readiness instead.
        :return: ``True`` if the server (or model) is ready.
        """
        return self._run(self._async.is_ready(model))

    def wait_for_ready(self, model: str | None = None, *, timeout: float = 300.0, interval: float = 2.0) -> None:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.wait_for_ready`.

        :param model: Model name to wait for; when omitted, waits for overall server readiness.
        :param timeout: Maximum time to wait, in seconds.
        :param interval: Delay between readiness checks, in seconds.
        :raises ModelNotReadyError: If the deadline passes without the target becoming ready.
        """
        self._run(self._async.wait_for_ready(model, timeout=timeout, interval=interval))

    def model_metadata(self, model: str | None = None) -> ModelMetadata:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.model_metadata`.

        :param model: Model name; the configured default model when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelMetadata`.
        """
        return self._run(self._async.model_metadata(model))

    def model_config(self, model: str | None = None) -> ModelConfig:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.model_config`.

        :param model: Model name; the configured default model when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelConfig`.
        """
        return self._run(self._async.model_config(model))

    def list_models(self) -> list[RepositoryModel]:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.list_models`.

        :return: One :class:`~triton_sdk.core.models.RepositoryModel` per model/version.
        """
        return self._run(self._async.list_models())

    @property
    def chat_template(self) -> ChatTemplate:
        """The chat template used to render messages.

        :return: The bound :class:`~triton_sdk.chat.ChatTemplate`.
        """
        return self._async.chat_template

    def resolve_model(self, model: str | None) -> str:
        """Resolve the model name to use for a call, falling back to the configured default.

        :param model: Model name passed by the caller, or ``None`` to use the default.
        :return: The resolved model name.
        :raises ValueError: If ``model`` is ``None`` and no default model is configured.
        """
        return self._async.resolve_model(model)

    def render_chat(self, messages: list[ChatMessage]) -> str:
        """Render a conversation into a raw prompt using the bound chat template.

        :param messages: The conversation, in order.
        :return: The rendered prompt string.
        """
        return self._async.render_chat(messages)

    def generate(self, prompt: str, params: SamplingParams | None = None, **kwargs: Any) -> GenerateResponse:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.generate`.

        :param prompt: Raw prompt text.
        :param params: Sampling parameters; see :meth:`AsyncTritonLLMClient.generate`.
        :param kwargs: Extra keyword arguments forwarded to
            :meth:`AsyncTritonLLMClient.generate`.
        :return: The assembled :class:`~triton_sdk.core.models.GenerateResponse`.
        """
        return self._run(self._async.generate(prompt, params, **kwargs))

    def stream(self, prompt: str, params: SamplingParams | None = None, **kwargs: Any) -> Iterator[GenerateChunk]:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.stream`.

        Breaking out of the ``for`` loop cancels the request, same as the async version.

        :param prompt: Raw prompt text.
        :param params: Sampling parameters; see :meth:`AsyncTritonLLMClient.stream`.
        :param kwargs: Extra keyword arguments forwarded to :meth:`AsyncTritonLLMClient.stream`.
        :return: A synchronous iterator of :class:`~triton_sdk.core.models.GenerateChunk`.
        """
        return self._runner.iterate(self._async.stream(prompt, params, **kwargs))  # type: ignore[arg-type]

    def chat(
        self, messages: list[ChatMessage], params: SamplingParams | None = None, **kwargs: Any
    ) -> GenerateResponse:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.chat`.

        :param messages: The conversation, in order.
        :param params: Sampling parameters; see :meth:`AsyncTritonLLMClient.chat`.
        :param kwargs: Extra keyword arguments forwarded to :meth:`AsyncTritonLLMClient.chat`.
        :return: The assembled :class:`~triton_sdk.core.models.GenerateResponse`.
        """
        return self._run(self._async.chat(messages, params, **kwargs))

    def chat_stream(
        self, messages: list[ChatMessage], params: SamplingParams | None = None, **kwargs: Any
    ) -> Iterator[GenerateChunk]:
        """Synchronous equivalent of :meth:`AsyncTritonLLMClient.chat_stream`.

        :param messages: The conversation, in order.
        :param params: Sampling parameters; see :meth:`AsyncTritonLLMClient.chat_stream`.
        :param kwargs: Extra keyword arguments forwarded to
            :meth:`AsyncTritonLLMClient.chat_stream`.
        :return: A synchronous iterator of :class:`~triton_sdk.core.models.GenerateChunk`.
        """
        return self._runner.iterate(self._async.chat_stream(messages, params, **kwargs))  # type: ignore[arg-type]
