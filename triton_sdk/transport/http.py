"""HTTP transport on top of httpx.

- infer()        -> POST /v2/models/{m}/infer (KServe v2 tensors JSON)
- stream_infer() -> POST /v2/models/{m}/generate_stream (generate extension, SSE). The plain /infer endpoint
                    cannot stream decoupled models over HTTP, so streaming always goes through generate_stream.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import httpx

from ..core.constants import Endpoint, versioned
from ..core.models import InferInput, InferResult, ModelConfig, ModelMetadata, RepositoryModel, ServerMetadata
from ..core.utils import is_loopback_url, normalize_url
from ..exceptions import TritonConnectionError, TritonTimeoutError, from_http_status
from .base import TransportInterface
from .kserve_json import (
    build_generate_body,
    build_infer_body,
    mark_final,
    parse_generate_event,
    parse_infer_response,
)
from .sse import parse_sse
from .tls import TLSConfig


logger = logging.getLogger(__name__)


class HttpTransport(TransportInterface):
    """:class:`~triton_sdk.transport.base.TransportInterface` implementation over HTTP (httpx)."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        connect_timeout: float = 5.0,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
        tls: TLSConfig | None = None,
    ) -> None:
        """Create the underlying httpx.AsyncClient.

        :param url: Server address, either ``host:port`` or a full URL; ``https://`` enables TLS.
        :param timeout: Default per-call timeout, in seconds.
        :param connect_timeout: Timeout for establishing the TCP connection, in seconds.
        :param verify_ssl: Whether to verify the server certificate; ignored when ``tls`` is given.
        :param headers: Extra headers sent with every request (e.g. ``Authorization``).
        :param tls: TLS configuration; built from ``verify_ssl`` when omitted.
        """
        tls = (tls or TLSConfig(verify=verify_ssl)).with_url(url)
        self._base_url = normalize_url(url, secure=tls.enabled)
        self._default_timeout = timeout
        self._connect_timeout = connect_timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout(None),
            verify=tls.httpx_verify(),
            cert=tls.httpx_cert(),
            headers=headers or {},
            # Loopback through HTTP_PROXY never reaches Triton (503 from the proxy, empty server logs).
            trust_env=not is_loopback_url(self._base_url),
        )

    # --- helpers -----------------------------------------------------------
    def _timeout(self, timeout: float | None) -> httpx.Timeout:
        """Build an httpx.Timeout for one call, falling back to the client default.

        :param timeout: Per-call timeout, in seconds; the default is used when ``None``.
        :return: The resulting ``httpx.Timeout``.
        """
        return httpx.Timeout(timeout or self._default_timeout, connect=self._connect_timeout)

    async def _request(self, method: str, path: str, *, model: str | None = None, **kwargs: Any) -> httpx.Response:
        """Issue one HTTP request and translate transport/HTTP errors into TritonError.

        :param method: HTTP method, e.g. ``"GET"`` or ``"POST"``.
        :param path: Request path, relative to the client base URL.
        :param model: Model name to attach to any resulting error.
        :param kwargs: Extra keyword arguments forwarded to ``httpx.AsyncClient.request``.
        :return: The successful ``httpx.Response``.
        :raises TritonTimeoutError: On a request timeout.
        :raises TritonConnectionError: On a connection failure.
        :raises TritonError: If the server returns an error status.
        """
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise TritonTimeoutError(str(exc), model=model) from exc
        except httpx.TransportError as exc:
            raise TritonConnectionError(str(exc), model=model) from exc
        if response.is_error:
            raise from_http_status(response.status_code, _error_message(response), model=model)
        return response

    # --- health / metadata -------------------------------------------------
    async def server_live(self) -> bool:
        """Check whether the server process is up.

        :return: True if the liveness endpoint returns HTTP 200.
        """
        response = await self._client.get(Endpoint.SERVER_LIVE)
        return response.status_code == httpx.codes.OK

    async def server_ready(self) -> bool:
        """Check whether the server is ready to accept inference requests.

        :return: True if the readiness endpoint returns HTTP 200.
        """
        response = await self._client.get(Endpoint.SERVER_READY)
        return response.status_code == httpx.codes.OK

    async def model_ready(self, model: str, version: str | None = None) -> bool:
        """Check whether a specific model (version) is loaded and ready.

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: True if the model readiness endpoint returns HTTP 200.
        """
        response = await self._client.get(versioned(Endpoint.MODEL_READY, model, version))
        return response.status_code == httpx.codes.OK

    async def server_metadata(self) -> ServerMetadata:
        """Fetch server name, version and supported extensions.

        :return: The parsed ``ServerMetadata``.
        :raises TritonError: If the request fails.
        """
        response = await self._request("GET", Endpoint.SERVER_METADATA)
        return ServerMetadata.model_validate(response.json())

    async def model_metadata(self, model: str, version: str | None = None) -> ModelMetadata:
        """Fetch a model's declared inputs, outputs and available versions.

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: The parsed ``ModelMetadata``.
        :raises ModelNotFoundError: If the model or version is unknown.
        """
        response = await self._request("GET", versioned(Endpoint.MODEL_METADATA, model, version), model=model)
        return ModelMetadata.model_validate(response.json())

    async def model_config(self, model: str, version: str | None = None) -> ModelConfig:
        """Fetch a model's config.pbtxt (backend, batching, decoupled flag, tensor schema).

        :param model: Model name.
        :param version: Model version; the server default version when omitted.
        :return: The parsed ``ModelConfig``.
        :raises ModelNotFoundError: If the model or version is unknown.
        """
        response = await self._request("GET", versioned(Endpoint.MODEL_CONFIG, model, version), model=model)
        return ModelConfig.model_validate(response.json())

    async def repository_index(self) -> list[RepositoryModel]:
        """List every model the server repository knows about, with its load state.

        :return: One ``RepositoryModel`` per model/version.
        """
        response = await self._request("POST", Endpoint.REPOSITORY_INDEX, json={})
        return [RepositoryModel.model_validate(item) for item in response.json()]

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
        """Run one unary inference call via POST /v2/models/{model}/infer.

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
        body = build_infer_body(inputs, outputs=outputs, request_id=request_id, parameters=parameters)
        response = await self._request(
            "POST",
            versioned(Endpoint.MODEL_INFER, model, version),
            model=model,
            json=body,
            timeout=self._timeout(timeout),
        )
        return parse_infer_response(response.json(), model=model)

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
        """Stream via generate_stream.

        timeout is the maximum idle time between two SSE events, not the total stream duration.
        The last result is always flagged with triton_final_response, even if the server omitted it.
        Closing the iterator (break / aclose) drops the connection, which is the only cancel HTTP offers.

        :param model: Model name.
        :param inputs: Input tensors to send.
        :param outputs: Ignored; the generate extension does not support output selection.
        :param version: Model version; the server default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param parameters: Extra request-level parameters understood by the backend.
        :param timeout: Maximum idle time between two SSE events, in seconds.
        :return: An async generator of ``InferResult``.
        :raises TritonError: On any transport, timeout or inference failure.
        """
        body = build_generate_body(inputs, request_id=request_id, parameters=parameters)
        pending: InferResult | None = None
        async for event in self._generate_stream_events(model, body, version=version, timeout=timeout):
            result = parse_generate_event(event, model=model)
            if pending is not None:
                yield pending
            pending = result
        if pending is not None:
            yield mark_final(pending)

    # --- generate endpoints (LLM-friendly JSON) -------------------------------
    async def generate(
        self, model: str, body: dict[str, Any], *, version: str | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        """Issue POST /v2/models/{model}/generate with a flat JSON body.

        :param model: Model name.
        :param body: Flat request body, as produced by ``build_generate_body``.
        :param version: Model version; the server default version when omitted.
        :param timeout: Per-call timeout, in seconds; the transport default when omitted.
        :return: The decoded JSON response body.
        :raises TritonError: On any transport, timeout or inference failure.
        """
        response = await self._request(
            "POST",
            versioned(Endpoint.MODEL_GENERATE, model, version),
            model=model,
            json=body,
            timeout=self._timeout(timeout),
        )
        return response.json()

    async def generate_stream(
        self, model: str, body: dict[str, Any], *, version: str | None = None, timeout: float | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Issue POST /v2/models/{model}/generate_stream and yield each decoded SSE event.

        :param model: Model name.
        :param body: Flat request body, as produced by ``build_generate_body``.
        :param version: Model version; the server default version when omitted.
        :param timeout: Maximum idle time between two SSE events, in seconds.
        :return: An async iterator of decoded event payloads.
        :raises TritonError: On any transport, timeout or inference failure.
        """
        async for event in self._generate_stream_events(model, body, version=version, timeout=timeout):
            yield event

    async def _generate_stream_events(
        self, model: str, body: dict[str, Any], *, version: str | None, timeout: float | None
    ) -> AsyncIterator[dict[str, Any]]:
        """Open the generate_stream connection and yield its parsed SSE events.

        Shared by :meth:`stream_infer` and :meth:`generate_stream`.

        :param model: Model name.
        :param body: Flat request body to send.
        :param version: Model version; the server default version when omitted.
        :param timeout: Maximum idle time between two SSE events, in seconds.
        :return: An async iterator of decoded event payloads.
        :raises TritonError: On any transport, timeout or inference failure.
        """
        path = versioned(Endpoint.MODEL_GENERATE_STREAM, model, version)
        try:
            async with self._client.stream("POST", path, json=body, timeout=self._timeout(timeout)) as response:
                if response.is_error:
                    await response.aread()
                    raise from_http_status(response.status_code, _error_message(response), model=model)
                async for event in parse_sse(response.aiter_lines()):
                    yield event
        except httpx.TimeoutException as exc:
            raise TritonTimeoutError(str(exc), model=model) from exc
        except httpx.TransportError as exc:
            raise TritonConnectionError(str(exc), model=model) from exc

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient. Safe to call more than once."""
        await self._client.aclose()


def _error_message(response: httpx.Response) -> str:
    """Extract a human-readable error message from an error response.

    :param response: The error response to inspect.
    :return: The ``"error"`` field of the JSON body when present, otherwise the raw response text.
    """
    try:
        return str(response.json().get("error", response.text))
    except ValueError:
        return response.text
