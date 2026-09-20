"""HTTP client for Triton's OpenAI-compatible frontend (server repo: python/openai, default port 9000).

Endpoints: POST /v1/chat/completions, POST /v1/completions (JSON or SSE with a terminating `data: [DONE]`),
GET /v1/models, GET /health/ready. The chat template is applied server-side (--tokenizer), usage comes back ready.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..core.utils import is_loopback_url, normalize_url
from ..exceptions import TritonConnectionError, TritonTimeoutError, from_http_status
from ..transport.sse import parse_sse
from ..transport.tls import TLSConfig


CHAT_COMPLETIONS = "/v1/chat/completions"
COMPLETIONS = "/v1/completions"
MODELS = "/v1/models"
READY = "/health/ready"


class AsyncOpenAIFrontend:
    """Thin async HTTP client for Triton's built-in OpenAI-compatible frontend."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 60.0,
        stream_timeout: float = 300.0,
        connect_timeout: float = 5.0,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
        tls: TLSConfig | None = None,
    ) -> None:
        """Create the underlying httpx.AsyncClient.

        :param url: Frontend address, either ``host:port`` or a full URL; ``https://`` enables TLS.
        :param timeout: Default timeout for non-streaming requests, in seconds.
        :param stream_timeout: Timeout applied to streaming requests, in seconds.
        :param connect_timeout: Timeout for establishing the TCP connection, in seconds.
        :param verify_ssl: Whether to verify the server certificate; ignored when ``tls`` is given.
        :param headers: Extra headers sent with every request (e.g. ``Authorization``).
        :param tls: TLS configuration; built from ``verify_ssl`` when omitted.
        """
        tls = (tls or TLSConfig(verify=verify_ssl)).with_url(url)
        self._stream_timeout = httpx.Timeout(stream_timeout, connect=connect_timeout)
        base_url = normalize_url(url, secure=tls.enabled)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            verify=tls.httpx_verify(),
            cert=tls.httpx_cert(),
            headers=headers or {},
            # Loopback through HTTP_PROXY never reaches Triton (503 from the proxy, empty server logs).
            trust_env=not is_loopback_url(base_url),
        )

    async def ready(self) -> bool:
        """Check whether the frontend is ready.

        :return: ``True`` if ``GET /health/ready`` returns HTTP 200; ``False`` on any
            connection failure.
        """
        try:
            response = await self._client.get(READY)
        except httpx.TransportError:
            return False
        return response.status_code == httpx.codes.OK

    async def models(self) -> dict[str, Any]:
        """Fetch the models listing.

        :return: The decoded JSON body of ``GET /v1/models``.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        return (await self._post_or_get("GET", MODELS)).json()

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Issue a non-streaming chat completion request.

        :param body: Request body, matching the OpenAI chat completions schema.
        :return: The decoded JSON body of ``POST /v1/chat/completions``.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        return (await self._post_or_get("POST", CHAT_COMPLETIONS, json={**body, "stream": False})).json()

    def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Issue a streaming chat completion request.

        :param body: Request body, matching the OpenAI chat completions schema.
        :return: An async iterator of decoded SSE event payloads.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        return self._stream(CHAT_COMPLETIONS, {**body, "stream": True})

    async def completions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Issue a non-streaming legacy completion request.

        :param body: Request body, matching the OpenAI completions schema.
        :return: The decoded JSON body of ``POST /v1/completions``.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        return (await self._post_or_get("POST", COMPLETIONS, json={**body, "stream": False})).json()

    def completions_stream(self, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Issue a streaming legacy completion request.

        :param body: Request body, matching the OpenAI completions schema.
        :return: An async iterator of decoded SSE event payloads.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        return self._stream(COMPLETIONS, {**body, "stream": True})

    async def close(self) -> None:
        """Close the underlying httpx.AsyncClient. Safe to call more than once."""
        await self._client.aclose()

    # --- helpers -----------------------------------------------------------------
    async def _post_or_get(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Issue one request and translate transport/HTTP errors into TritonError.

        :param method: HTTP method, e.g. ``"GET"`` or ``"POST"``.
        :param path: Request path, relative to the client base URL.
        :param kwargs: Extra keyword arguments forwarded to ``httpx.AsyncClient.request``; a
            ``"json"`` body's ``"model"`` key, if present, is attached to any resulting error.
        :return: The successful ``httpx.Response``.
        :raises TritonTimeoutError: On a request timeout.
        :raises TritonConnectionError: On a connection failure.
        :raises TritonError: If the server returns an error status.
        """
        model = (kwargs.get("json") or {}).get("model")
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise TritonTimeoutError(str(exc), model=model) from exc
        except httpx.TransportError as exc:
            raise TritonConnectionError(str(exc), model=model) from exc
        if response.is_error:
            raise from_http_status(response.status_code, _error_message(response), model=model)
        return response

    async def _stream(self, path: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Open a streaming POST request and yield its parsed SSE events.

        :param path: Request path, relative to the client base URL.
        :param body: JSON request body to send.
        :return: An async iterator of decoded event payloads.
        :raises TritonError: On any transport, timeout or HTTP error.
        """
        model = body.get("model")
        try:
            async with self._client.stream("POST", path, json=body, timeout=self._stream_timeout) as response:
                if response.is_error:
                    await response.aread()
                    raise from_http_status(response.status_code, _error_message(response), model=model)
                async for event in parse_sse(response.aiter_lines()):
                    yield event
        except httpx.TimeoutException as exc:
            raise TritonTimeoutError(str(exc), model=model) from exc
        except httpx.TransportError as exc:
            raise TritonConnectionError(str(exc), model=model) from exc


def _error_message(response: httpx.Response) -> str:
    """Extract a human-readable error message from an error response.

    :param response: The error response to inspect.
    :return: The ``detail``/``error`` field of the JSON body when present (unwrapping a nested
        ``message`` key), otherwise the raw response text.
    """
    try:
        payload = response.json()
    except ValueError:
        return response.text
    detail = payload.get("detail") or payload.get("error") or payload
    if isinstance(detail, dict):
        detail = detail.get("message", detail)
    return str(detail)
