"""gRPC transport on top of grpc.aio. Requires the grpc extra and generated stubs in transport/protos.

- infer()        -> GRPCInferenceService.ModelInfer (unary)
- stream_infer() -> GRPCInferenceService.ModelStreamInfer (bidi). One request is written, responses are read until
                    triton_final_response. Closing the generator cancels the call, which stops generation server-side.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypeVar

from ..core.constants import FINAL_RESPONSE_PARAM
from ..core.models import InferInput, InferResult, ModelConfig, ModelMetadata, RepositoryModel, ServerMetadata
from ..core.optional import require_import
from ..exceptions import (
    InferenceError,
    StreamCancelledError,
    TritonTimeoutError,
    from_grpc_status,
)
from .base import TransportInterface
from .tls import TLSConfig, grpc_target


grpc = require_import("grpc", "grpc")  # pragma: no cover - exercised only without the extra

from . import grpc_codec  # noqa: E402 - depends on grpc, imported above
from .protos import grpc_service_pb2 as pb  # noqa: E402
from .protos import grpc_service_pb2_grpc as pb_grpc  # noqa: E402


logger = logging.getLogger(__name__)

T = TypeVar("T")


class GrpcTransport(TransportInterface):
    """:class:`~triton_sdk.transport.base.TransportInterface` implementation over gRPC (grpc.aio)."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        tls: TLSConfig | None = None,
        metadata: dict[str, str] | None = None,
        max_message_length: int = 64 * 1024 * 1024,
        channel: Any = None,
    ) -> None:
        """Create the underlying grpc.aio channel and stub.

        :param url: Server address, either ``host:port`` or a URL with a ``grpc://``/``grpcs://``
            scheme (any scheme is stripped).
        :param timeout: Default per-call timeout, in seconds.
        :param tls: TLS configuration; plaintext when omitted, unless ``url`` uses ``grpcs://``.
        :param metadata: Extra gRPC metadata sent with every call (e.g. ``Authorization``); keys
            are lowercased, as gRPC requires.
        :param max_message_length: Maximum send/receive message size, in bytes.
        :param channel: A pre-built ``grpc.aio.Channel`` to reuse instead of creating one
            (primarily for tests).
        """
        tls = (tls or TLSConfig()).with_url(url)
        self._url = grpc_target(url)
        self._default_timeout = timeout
        # gRPC metadata keys must be lowercase (Authorization -> authorization)
        self._metadata = tuple((key.lower(), value) for key, value in (metadata or {}).items())
        options = [
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            *tls.grpc_options(),
        ]
        if channel is not None:
            self._channel = channel
        elif tls.enabled:
            self._channel = grpc.aio.secure_channel(self._url, tls.grpc_credentials(), options=options)
        else:
            self._channel = grpc.aio.insecure_channel(self._url, options=options)
        self._stub = pb_grpc.GRPCInferenceServiceStub(self._channel)

    # --- helpers -----------------------------------------------------------
    async def _call(
        self,
        method: Callable[..., Awaitable[T]],
        request: Any,
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> T:
        """Invoke a unary RPC and translate grpc errors into TritonError.

        :param method: Bound stub method to call, e.g. ``self._stub.ServerReady``.
        :param request: Protobuf request message to send.
        :param model: Model name to attach to any resulting error.
        :param timeout: Per-call timeout, in seconds; the transport default when omitted.
        :return: The RPC's response message.
        :raises TritonError: If the RPC fails, mapped from the gRPC status code.
        """
        try:
            return await method(request, timeout=timeout or self._default_timeout, metadata=self._metadata)
        except grpc.aio.AioRpcError as exc:
            raise from_grpc_status(exc.code().name, exc.details() or "", model=model) from exc

    # --- health / metadata -------------------------------------------------
    async def server_live(self) -> bool:
        """Check whether the server process is up.

        :return: The value of the response's ``live`` field.
        """
        response = await self._call(self._stub.ServerLive, pb.ServerLiveRequest())
        return bool(response.live)

    async def server_ready(self) -> bool:
        """Check whether the server is ready to accept inference requests.

        :return: The value of the response's ``ready`` field.
        """
        response = await self._call(self._stub.ServerReady, pb.ServerReadyRequest())
        return bool(response.ready)

    async def model_ready(self, model: str, version: str | None = None) -> bool:
        """Check whether a specific model (version) is loaded and ready.

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: The value of the response's ``ready`` field.
        """
        request = pb.ModelReadyRequest(name=model, version=version or "")
        response = await self._call(self._stub.ModelReady, request, model=model)
        return bool(response.ready)

    async def server_metadata(self) -> ServerMetadata:
        """Fetch server name, version and supported extensions.

        :return: The parsed ``ServerMetadata``.
        :raises TritonError: If the RPC fails.
        """
        response = await self._call(self._stub.ServerMetadata, pb.ServerMetadataRequest())
        return grpc_codec.parse_server_metadata(response)

    async def model_metadata(self, model: str, version: str | None = None) -> ModelMetadata:
        """Fetch a model's declared inputs, outputs and available versions.

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: The parsed ``ModelMetadata``.
        :raises ModelNotFoundError: If the model or version is unknown.
        """
        request = pb.ModelMetadataRequest(name=model, version=version or "")
        response = await self._call(self._stub.ModelMetadata, request, model=model)
        return grpc_codec.parse_model_metadata(response)

    async def model_config(self, model: str, version: str | None = None) -> ModelConfig:
        """Fetch a model's config.pbtxt (backend, batching, decoupled flag, tensor schema).

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: The parsed ``ModelConfig``.
        :raises ModelNotFoundError: If the model or version is unknown.
        """
        request = pb.ModelConfigRequest(name=model, version=version or "")
        response = await self._call(self._stub.ModelConfig, request, model=model)
        return grpc_codec.parse_model_config(response)

    async def repository_index(self) -> list[RepositoryModel]:
        """List every model the server repository knows about, with its load state.

        :return: One ``RepositoryModel`` per model/version.
        """
        response = await self._call(self._stub.RepositoryIndex, pb.RepositoryIndexRequest())
        return grpc_codec.parse_repository_index(response)

    # --- inference ---------------------------------------------------------
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
        """Run one unary inference call via the ModelInfer RPC.

        :param model: Model name.
        :param inputs: Input tensors to send.
        :param outputs: Names of output tensors to request; the server defaults when omitted.
        :param version: Model version; the server default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param parameters: Extra request-level parameters understood by the backend.
        :param timeout: Per-call timeout, in seconds; the transport default when omitted.
        :return: The parsed ``InferResult``.
        :raises TritonError: On any transport, timeout or inference failure.
        """
        request = grpc_codec.build_infer_request(
            model, inputs, outputs=outputs, version=version, request_id=request_id, parameters=parameters
        )
        response = await self._call(self._stub.ModelInfer, request, model=model, timeout=timeout)
        return grpc_codec.parse_infer_response(response)

    async def stream_infer(
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
        """Run a bidirectional ModelStreamInfer call.

        timeout is the maximum idle time between two responses (same semantics as HttpTransport).
        Breaking out of the loop cancels the RPC; Triton then stops generating for this request.

        :param model: Model name.
        :param inputs: Input tensors to send.
        :param outputs: Names of output tensors to request; the server defaults when omitted.
        :param version: Model version; the server default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param parameters: Extra request-level parameters understood by the backend.
        :param timeout: Maximum idle time between two responses, in seconds.
        :return: An async generator of ``InferResult``.
        :raises TritonTimeoutError: If no response arrives within ``timeout``.
        :raises InferenceError: If the server reports an error mid-stream.
        :raises StreamCancelledError: If the stream is cancelled.
        :raises TritonError: On any other RPC failure.
        """
        request = grpc_codec.build_infer_request(
            model, inputs, outputs=outputs, version=version, request_id=request_id, parameters=parameters
        )
        idle_timeout = timeout or self._default_timeout
        call = self._stub.ModelStreamInfer(metadata=self._metadata)
        finished = False
        try:
            await call.write(request)
            await call.done_writing()
            while True:
                try:
                    message = await asyncio.wait_for(call.read(), idle_timeout)
                except TimeoutError as exc:
                    msg = f"no response for {idle_timeout}s"
                    raise TritonTimeoutError(msg, model=model) from exc
                if message is grpc.aio.EOF:
                    finished = True
                    return
                if message.error_message:
                    raise InferenceError(message.error_message, model=model)
                result = grpc_codec.parse_infer_response(message.infer_response)
                yield result
                if result.parameters.get(FINAL_RESPONSE_PARAM):
                    finished = True
                    return
        except grpc.aio.AioRpcError as exc:
            if exc.code() == grpc.StatusCode.CANCELLED:
                raise StreamCancelledError(exc.details() or "cancelled", model=model) from exc
            raise from_grpc_status(exc.code().name, exc.details() or "", model=model) from exc
        finally:
            if not finished:
                logger.debug("cancelling stream for model %s", model)
                call.cancel()

    async def close(self) -> None:
        """Close the underlying grpc.aio channel. Safe to call more than once."""
        await self._channel.close()
