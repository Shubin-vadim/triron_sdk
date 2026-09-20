"""TLSConfig mapping plus real TLS / mTLS handshakes against an in-process grpc.aio server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from triton_sdk.settings import TritonSettings
from triton_sdk.transport.http import HttpTransport
from triton_sdk.transport.tls import TLSConfig, grpc_target


CERTS = Path(__file__).parents[2] / "fixtures" / "certs"


# --- pure mapping ---------------------------------------------------------------------


def test_from_settings_and_url_scheme() -> None:
    settings = TritonSettings(  # type: ignore[call-arg]
        URL="triton.internal:8001",
        SSL=False,
        SSL_ROOT_CERT=CERTS / "ca.crt",
        SSL_CLIENT_CERT=CERTS / "client.crt",
        SSL_CLIENT_KEY=CERTS / "client.key",
        SSL_SERVER_NAME_OVERRIDE="localhost",
        _env_file=None,
    )
    tls = TLSConfig.from_settings(settings)
    assert tls.enabled is False
    assert tls.with_url("grpcs://triton.internal:8001").enabled is True
    assert tls.with_url("https://triton.internal:8000").enabled is True
    assert tls.with_url("triton.internal:8001").enabled is False
    assert tls.httpx_verify() == str(CERTS / "ca.crt")
    assert tls.httpx_cert() == (str(CERTS / "client.crt"), str(CERTS / "client.key"))
    assert tls.grpc_options() == [("grpc.ssl_target_name_override", "localhost")]


def test_httpx_mapping_defaults() -> None:
    assert TLSConfig().httpx_verify() is True
    assert TLSConfig(verify=False).httpx_verify() is False
    assert TLSConfig().httpx_cert() is None
    assert TLSConfig(client_cert=CERTS / "client.crt").httpx_cert() == (
        str(CERTS / "client.crt"),
        str(CERTS / "client.crt"),
    )


@pytest.mark.parametrize(
    ("url", "target"),
    [("localhost:8001", "localhost:8001"), ("grpcs://h:1/", "h:1"), ("https://h:2", "h:2"), ("grpc://h:3", "h:3")],
)
def test_grpc_target_strips_scheme(url: str, target: str) -> None:
    assert grpc_target(url) == target


async def test_http_transport_switches_to_https_when_tls_enabled() -> None:
    async with HttpTransport("triton:8000", tls=TLSConfig(enabled=True)) as plain_host:
        assert str(plain_host._client.base_url) == "https://triton:8000"
    async with HttpTransport("https://triton:8443") as scheme_only:
        assert str(scheme_only._client.base_url) == "https://triton:8443"
    async with HttpTransport("triton:8000") as insecure:
        assert str(insecure._client.base_url) == "http://triton:8000"


# --- live handshakes ------------------------------------------------------------------

grpc = pytest.importorskip("grpc", reason="grpc extra not installed")

from triton_sdk.exceptions import TritonConnectionError  # noqa: E402
from triton_sdk.transport.grpc import GrpcTransport  # noqa: E402
from triton_sdk.transport.protos import grpc_service_pb2_grpc as pb_grpc  # noqa: E402

from .test_grpc import FakeTriton  # noqa: E402


def _server_credentials(*, require_client_cert: bool) -> object:
    return grpc.ssl_server_credentials(
        [((CERTS / "server.key").read_bytes(), (CERTS / "server.crt").read_bytes())],
        root_certificates=(CERTS / "ca.crt").read_bytes() if require_client_cert else None,
        require_client_auth=require_client_cert,
    )


async def _start(*, mtls: bool) -> tuple[object, str]:
    server = grpc.aio.server()
    pb_grpc.add_GRPCInferenceServiceServicer_to_server(FakeTriton(), server)
    port = server.add_secure_port("localhost:0", _server_credentials(require_client_cert=mtls))
    await server.start()
    return server, f"localhost:{port}"


@pytest.fixture
async def tls_server() -> AsyncIterator[str]:
    server, address = await _start(mtls=False)
    try:
        yield address
    finally:
        await server.stop(grace=None)  # type: ignore[attr-defined]


@pytest.fixture
async def mtls_server() -> AsyncIterator[str]:
    server, address = await _start(mtls=True)
    try:
        yield address
    finally:
        await server.stop(grace=None)  # type: ignore[attr-defined]


async def test_grpc_tls_with_custom_ca(tls_server: str) -> None:
    async with GrpcTransport(
        tls_server, timeout=5, tls=TLSConfig(enabled=True, root_cert=CERTS / "ca.crt")
    ) as transport:
        assert await transport.server_ready() is True


async def test_grpc_tls_via_grpcs_scheme(tls_server: str) -> None:
    async with GrpcTransport(
        f"grpcs://{tls_server}", timeout=5, tls=TLSConfig(root_cert=CERTS / "ca.crt")
    ) as transport:
        assert await transport.server_ready() is True


async def test_grpc_plaintext_client_cannot_reach_tls_server(tls_server: str) -> None:
    async with GrpcTransport(tls_server, timeout=2) as transport:
        with pytest.raises(TritonConnectionError):
            await transport.server_ready()


async def test_grpc_untrusted_ca_is_rejected(tls_server: str) -> None:
    # system roots do not know our test CA -> handshake fails -> UNAVAILABLE -> TritonConnectionError
    async with GrpcTransport(tls_server, timeout=2, tls=TLSConfig(enabled=True)) as transport:
        with pytest.raises(TritonConnectionError):
            await transport.server_ready()


async def test_grpc_mtls_requires_client_certificate(mtls_server: str) -> None:
    without_cert = TLSConfig(enabled=True, root_cert=CERTS / "ca.crt")
    async with GrpcTransport(mtls_server, timeout=2, tls=without_cert) as transport:
        with pytest.raises(TritonConnectionError):
            await transport.server_ready()

    with_cert = without_cert.model_copy(
        update={"client_cert": CERTS / "client.crt", "client_key": CERTS / "client.key"}
    )
    async with GrpcTransport(mtls_server, timeout=5, tls=with_cert) as transport:
        assert await transport.server_ready() is True


async def test_grpc_server_name_override(tls_server: str) -> None:
    # connect via 127.0.0.1 but validate the certificate against "localhost"
    host, port = tls_server.split(":")
    assert host == "localhost"
    tls = TLSConfig(enabled=True, root_cert=CERTS / "ca.crt", server_name_override="localhost")
    async with GrpcTransport(f"127.0.0.1:{port}", timeout=5, tls=tls) as transport:
        assert await transport.server_ready() is True
