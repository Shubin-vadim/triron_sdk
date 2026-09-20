"""Small helpers shared by transport and client layers."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from triton_sdk.core.constants import RETRY_STATUSES
from triton_sdk.exceptions import TritonConnectionError, TritonError


def normalize_url(url: str, *, secure: bool = False) -> str:
    """Turn a bare ``host:port`` or a full URL into a URL with an explicit scheme.

    :param url: Either ``host:port`` or a URL that already has a scheme.
    :param secure: When ``url`` has no scheme, use ``https`` instead of ``http``.
    :return: The URL with a scheme, and no trailing slash.
    """
    if "://" in url:
        return url.rstrip("/")
    scheme = "https" if secure else "http"
    return f"{scheme}://{url}"


def is_loopback_url(url: str) -> bool:
    """True when ``url`` (bare ``host:port`` or full URL) points at this machine's loopback.

    HTTP clients must not honour ``HTTP_PROXY`` for these hosts: a corporate or sandbox proxy
    answers 503 and Triton never sees the request.

    :param url: Either ``host:port`` or a URL that already has a scheme.
    :return: ``True`` for ``localhost``, ``*.localhost``, and IPv4/IPv6 loopback addresses.
    """
    host = (urlparse(normalize_url(url)).hostname or "").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_retryable(exc: BaseException) -> bool:
    """Decide whether an exception raised by a request is worth retrying.

    :param exc: The exception to inspect.
    :return: ``True`` for connection failures and for :class:`~triton_sdk.exceptions.TritonError`
        instances whose ``status_code`` is in :data:`~triton_sdk.core.constants.RETRY_STATUSES`.
    """
    if isinstance(exc, TritonConnectionError):
        return True
    return isinstance(exc, TritonError) and exc.status_code in RETRY_STATUSES


def build_retrying(retry_count: int, start_timeout: float) -> AsyncRetrying:
    """Build the retry policy used for non-streaming calls. Streams are never retried mid-flight.

    :param retry_count: Maximum number of attempts.
    :param start_timeout: Base delay, in seconds, for the exponential backoff.
    :return: A configured :class:`tenacity.AsyncRetrying` instance.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(retry_count),
        wait=wait_exponential(multiplier=start_timeout),
        retry=retry_if_exception(is_retryable),
        reraise=True,
    )


def drop_none(values: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose value is ``None``.

    :param values: Source mapping.
    :return: A new dict containing only the entries whose value is not ``None``.
    """
    return {key: value for key, value in values.items() if value is not None}


def encode_bytes_tensor(values: list[str]) -> bytes:
    """Serialize BYTES tensor content for ``raw_input_contents``.

    Each element is encoded as a little-endian uint32 length prefix followed by its UTF-8 payload,
    matching the wire format Triton expects for gRPC string tensors.

    :param values: String values to encode, in tensor order.
    :return: The concatenated binary payload.
    """
    out = bytearray()
    for value in values:
        payload = value.encode("utf-8")
        out += len(payload).to_bytes(4, "little") + payload
    return bytes(out)


def decode_bytes_tensor(raw: bytes) -> list[str]:
    """Decode a BYTES tensor payload produced by :func:`encode_bytes_tensor`.

    :param raw: Binary payload as received in ``raw_output_contents``.
    :return: The decoded string values, in their original order.
    """
    values: list[str] = []
    offset = 0
    while offset < len(raw):
        length = int.from_bytes(raw[offset : offset + 4], "little")
        offset += 4
        values.append(raw[offset : offset + length].decode("utf-8"))
        offset += length
    return values
