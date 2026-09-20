"""TLS settings shared by the HTTP, gRPC and OpenAI-frontend transports.

One object, three consumers:
- gRPC  -> grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
- httpx -> verify= (CA bundle path / False) and cert=(client cert, key) for HTTP and the OpenAI frontend
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict


if TYPE_CHECKING:
    from ..settings import TritonSettings


logger = logging.getLogger(__name__)

SECURE_SCHEMES: frozenset[str] = frozenset({"https", "grpcs"})


class TLSConfig(BaseModel):
    """TLS/mTLS settings, translated on demand into the shapes ``httpx`` and ``grpc`` expect."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    verify: bool = True  # httpx only; grpc always verifies the server certificate
    root_cert: Path | None = None  # CA bundle; system roots when None
    client_cert: Path | None = None  # mTLS
    client_key: Path | None = None
    server_name_override: str | None = None  # grpc.ssl_target_name_override, for certs that do not match the host

    @classmethod
    def from_settings(cls, settings: TritonSettings) -> TLSConfig:
        """Build a :class:`TLSConfig` from the ``SSL*`` fields of :class:`~triton_sdk.settings.TritonSettings`.

        :param settings: Settings instance to read TLS fields from.
        :return: The equivalent :class:`TLSConfig`.
        """
        return cls(
            enabled=settings.SSL,
            verify=settings.VERIFY_SSL,
            root_cert=settings.SSL_ROOT_CERT,
            client_cert=settings.SSL_CLIENT_CERT,
            client_key=settings.SSL_CLIENT_KEY,
            server_name_override=settings.SSL_SERVER_NAME_OVERRIDE,
        )

    def with_url(self, url: str) -> TLSConfig:
        """Turn TLS on when the URL scheme demands it, even if it was otherwise disabled.

        :param url: The target URL; a ``grpcs://`` or ``https://`` scheme forces TLS on.
        :return: This config unchanged, or a copy with :attr:`enabled` set to ``True``.
        """
        scheme = url.split("://", 1)[0].lower() if "://" in url else ""
        if scheme in SECURE_SCHEMES and not self.enabled:
            return self.model_copy(update={"enabled": True})
        return self

    # --- httpx ---------------------------------------------------------------------
    def httpx_verify(self) -> Any:
        """Build the value for httpx's ``verify=`` argument.

        :return: ``False`` when verification is disabled, the CA bundle path as ``str`` when
            :attr:`root_cert` is set, otherwise ``True`` (use system roots).
        """
        if not self.verify:
            return False
        return str(self.root_cert) if self.root_cert else True

    def httpx_cert(self) -> tuple[str, str] | None:
        """Build the value for httpx's ``cert=`` argument (client certificate for mTLS).

        :return: A ``(cert_path, key_path)`` tuple, or ``None`` when no client certificate is configured.
        """
        if self.client_cert is None:
            return None
        return (str(self.client_cert), str(self.client_key or self.client_cert))

    # --- grpc ----------------------------------------------------------------------
    def grpc_credentials(self) -> Any:
        """Build ``grpc.ChannelCredentials`` for a secure channel.

        :return: The result of ``grpc.ssl_channel_credentials()`` built from this config.
        """
        import grpc

        if not self.verify:
            logger.warning("VERIFY_SSL=false has no effect on gRPC: the server certificate is always verified")
        return grpc.ssl_channel_credentials(
            root_certificates=_read(self.root_cert),
            private_key=_read(self.client_key or self.client_cert if self.client_cert else None),
            certificate_chain=_read(self.client_cert),
        )

    def grpc_options(self) -> list[tuple[str, Any]]:
        """Build extra ``grpc.aio`` channel options implied by this config.

        :return: A list with the ``grpc.ssl_target_name_override`` option when
            :attr:`server_name_override` is set, otherwise an empty list.
        """
        if self.server_name_override:
            return [("grpc.ssl_target_name_override", self.server_name_override)]
        return []


def _read(path: Path | None) -> bytes | None:
    """Read a PEM file's bytes, if a path was given.

    :param path: File path to read, or ``None``.
    :return: The file's contents, or ``None`` when ``path`` is ``None``.
    """
    return path.read_bytes() if path else None


def grpc_target(url: str) -> str:
    """Strip any URL scheme so the result is a bare ``host:port`` for gRPC.

    Accepts values copied from HTTP-style configuration (``grpc://``, ``grpcs://``, ``http://``,
    ``https://``) as well as plain ``host:port``.

    :param url: The URL or bare address to normalize.
    :return: ``host:port`` with no scheme and no trailing slash.
    """
    return url.split("://", 1)[1].rstrip("/") if "://" in url else url
