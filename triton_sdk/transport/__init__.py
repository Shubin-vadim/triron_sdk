from triton_sdk.transport.base import TransportInterface
from triton_sdk.transport.http import HttpTransport


__all__ = ["HttpTransport", "TransportInterface", "create_transport"]


def create_transport(protocol: str, url: str, **kwargs: object) -> TransportInterface:
    """Build a transport for the requested protocol.

    Used by the client layer. GrpcTransport is imported lazily to keep grpcio optional.

    :param protocol: Either "grpc" or "http" (anything else falls back to HTTP).
    :param url: Server address passed through to the transport constructor.
    :param kwargs: Extra keyword arguments passed through to the transport constructor.
    :return: A :class:`~triton_sdk.transport.grpc.GrpcTransport` for "grpc", otherwise an
        :class:`HttpTransport`.
    :raises MissingDependencyError: If protocol is "grpc" and the grpc extra is not installed.
    """
    if protocol == "grpc":
        from triton_sdk.transport.grpc import GrpcTransport

        return GrpcTransport(url, **kwargs)  # type: ignore[arg-type]
    return HttpTransport(url, **kwargs)  # type: ignore[arg-type]
