"""Transport contract. Implementations know nothing about LLMs, only about KServe v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Self

from triton_sdk.core.models import InferInput, InferResult, ModelConfig, ModelMetadata, RepositoryModel, ServerMetadata


class TransportInterface(ABC):
    """Async interface implemented by HttpTransport and GrpcTransport."""

    # --- health / metadata -------------------------------------------------
    @abstractmethod
    async def server_live(self) -> bool:
        """Check whether the server process is up.

        :return: ``True`` if the server responds to its liveness probe.
        """
        ...

    @abstractmethod
    async def server_ready(self) -> bool:
        """Check whether the server is ready to accept inference requests.

        :return: ``True`` if the server responds to its readiness probe.
        """
        ...

    @abstractmethod
    async def model_ready(self, model: str, version: str | None = None) -> bool:
        """Check whether a specific model (version) is loaded and ready.

        :param model: Model name.
        :param version: Model version; the server's default version when omitted.
        :return: ``True`` if the model is ready to serve requests.
        """
        ...

    @abstractmethod
    async def server_metadata(self) -> ServerMetadata:
        """Fetch server name, version and supported extensions.

        :return: The parsed :class:`~triton_sdk.core.models.ServerMetadata`.
        :raises triton_sdk.exceptions.TritonError: If the request fails.
        """
        ...

    @abstractmethod
    async def model_metadata(self, model: str, version: str | None = None) -> ModelMetadata:
        """Fetch a model's declared inputs, outputs and available versions.

        :param model: Model name.
        :param version: Model version; the server's default version when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelMetadata`.
        :raises triton_sdk.exceptions.ModelNotFoundError: If the model or version is unknown.
        """
        ...

    @abstractmethod
    async def model_config(self, model: str, version: str | None = None) -> ModelConfig:
        """Fetch a model's ``config.pbtxt`` (backend, batching, decoupled flag, tensor schema).

        :param model: Model name.
        :param version: Model version; the server's default version when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelConfig`.
        :raises triton_sdk.exceptions.ModelNotFoundError: If the model or version is unknown.
        """
        ...

    @abstractmethod
    async def repository_index(self) -> list[RepositoryModel]:
        """List every model the server's repository knows about, with its load state.

        :return: One :class:`~triton_sdk.core.models.RepositoryModel` per model/version.
        """
        ...

    # --- inference ---------------------------------------------------------
    @abstractmethod
    async def infer(
        self,
        model: str,
        inputs: list[InferInput],
        *,
        outputs: list[str] | None = None,
        version: str | None = None,
        request_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> InferResult:
        """Run one unary (non-streaming) inference call.

        :param model: Model name.
        :param inputs: Input tensors to send.
        :param outputs: Names of output tensors to request; the server's defaults when omitted.
        :param version: Model version; the server's default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param parameters: Extra request-level parameters understood by the backend.
        :param timeout: Per-call timeout, in seconds; the transport's default when omitted.
        :return: The server's response as a single :class:`~triton_sdk.core.models.InferResult`.
        :raises triton_sdk.exceptions.TritonError: On any transport, timeout or inference failure.
        """
        ...

    @abstractmethod
    def stream_infer(
        self,
        model: str,
        inputs: list[InferInput],
        *,
        outputs: list[str] | None = None,
        version: str | None = None,
        request_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AsyncGenerator[InferResult, None]:
        """Run a decoupled (streaming) inference call.

        Yields one :class:`~triton_sdk.core.models.InferResult` per server response. Closing the
        returned async generator early (e.g. via ``break`` in the caller's ``async for``) cancels
        the underlying request on the server.

        :param model: Model name.
        :param inputs: Input tensors to send.
        :param outputs: Names of output tensors to request; the server's defaults when omitted.
        :param version: Model version; the server's default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param parameters: Extra request-level parameters understood by the backend.
        :param timeout: Maximum idle time between two responses, in seconds.
        :return: An async generator of :class:`~triton_sdk.core.models.InferResult`.
        :raises triton_sdk.exceptions.TritonError: On any transport, timeout or inference failure.
        """
        ...

    # --- lifecycle ---------------------------------------------------------
    @abstractmethod
    async def close(self) -> None:
        """Release the underlying connection(s). Safe to call more than once."""
        ...

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        :return: This transport instance.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager, closing the transport.

        :param exc_info: Standard ``(exc_type, exc_value, traceback)`` tuple, unused.
        """
        await self.close()
