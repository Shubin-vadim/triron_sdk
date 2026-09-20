"""Error hierarchy. Transport layers translate httpx / grpc errors into these types."""

from __future__ import annotations


class TritonError(Exception):
    """Base class for every error raised across the public boundary."""

    def __init__(self, message: str, *, status_code: int | None = None, model: str | None = None) -> None:
        """Initialize the error.

        :param message: Human-readable description of what went wrong.
        :param status_code: HTTP status code (or its gRPC equivalent) that triggered this error, if any.
        :param model: Name of the model the request targeted, if known.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.model = model


class TritonConnectionError(TritonError):
    """Server unreachable / connection dropped."""


class TritonTimeoutError(TritonError):
    """Request or stream exceeded the configured timeout."""


class ModelNotFoundError(TritonError):
    """Model or version is unknown to the server (HTTP 404 / gRPC NOT_FOUND)."""


class ModelNotReadyError(TritonError):
    """Model exists but is still loading."""


class InferenceError(TritonError):
    """Server rejected or failed the inference request."""


class StreamCancelledError(TritonError):
    """Streaming request was cancelled by the caller."""


class BackendNotSupportedError(TritonError):
    """Model config does not match any registered backend adapter."""


class MissingDependencyError(TritonError):
    """Optional extra (grpc / chat) is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        """Initialize the error with an actionable install hint.

        :param package: Name of the missing third-party package (e.g. ``"grpcio"``).
        :param extra: Name of the triton-sdk extra that installs it (e.g. ``"grpc"``).
        """
        super().__init__(f"{package} is required: pip install triton-sdk[{extra}]")


def from_http_status(status_code: int, message: str, *, model: str | None = None) -> TritonError:
    """Map an HTTP status code to the closest :class:`TritonError` subclass.

    :param status_code: HTTP status code returned by the server.
    :param message: Error message to attach to the resulting exception.
    :param model: Name of the model the request targeted, if known.
    :return: A :class:`TritonError` instance of the most specific matching subclass.
    """
    if status_code == 404:  # noqa: PLR2004
        return ModelNotFoundError(message, status_code=status_code, model=model)
    if status_code == 400 and "not ready" in message.lower():  # noqa: PLR2004
        return ModelNotReadyError(message, status_code=status_code, model=model)
    if status_code in {408, 504}:
        return TritonTimeoutError(message, status_code=status_code, model=model)
    return InferenceError(message, status_code=status_code, model=model)


def from_grpc_status(code_name: str, message: str, *, model: str | None = None) -> TritonError:
    """Map a ``grpc.StatusCode`` name to the closest :class:`TritonError` subclass.

    :param code_name: Name of the gRPC status code (e.g. ``"NOT_FOUND"``, ``"UNAVAILABLE"``).
    :param message: Error message to attach to the resulting exception.
    :param model: Name of the model the request targeted, if known.
    :return: A :class:`TritonError` instance of the most specific matching subclass, or
        :class:`InferenceError` if the code has no dedicated mapping.
    """
    mapping: dict[str, type[TritonError]] = {
        "NOT_FOUND": ModelNotFoundError,
        "UNAVAILABLE": TritonConnectionError,
        "DEADLINE_EXCEEDED": TritonTimeoutError,
        "CANCELLED": StreamCancelledError,
    }
    return mapping.get(code_name, InferenceError)(message, model=model)
