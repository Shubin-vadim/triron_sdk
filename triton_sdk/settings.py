from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from triton_sdk.core.constants import BackendKind, Protocol


class TritonSettings(BaseSettings):
    """Connection defaults. Every field can be overridden per client instance."""

    model_config = SettingsConfigDict(env_prefix="TRITON_", env_file=".env", extra="ignore")

    URL: str = "localhost:8000"
    PROTOCOL: Protocol = Protocol.HTTP
    DEFAULT_MODEL: str | None = None
    BACKEND: BackendKind | None = None  # None -> auto-detect from model config

    TIMEOUT: float = 60.0
    STREAM_TIMEOUT: float = 300.0
    CONNECT_TIMEOUT: float = 5.0
    RETRY_COUNT: int = 3
    RETRY_START_TIMEOUT: float = 1.0
    VERIFY_SSL: bool = True  # httpx only; gRPC always verifies the server certificate

    # TLS. SSL=true (or a grpcs:// / https:// URL) turns it on for every transport.
    SSL: bool = False
    SSL_ROOT_CERT: Path | None = None  # CA bundle (PEM); system roots when unset
    SSL_CLIENT_CERT: Path | None = None  # mTLS client certificate (PEM)
    SSL_CLIENT_KEY: Path | None = None  # mTLS client key; defaults to SSL_CLIENT_CERT when the PEM holds both
    SSL_SERVER_NAME_OVERRIDE: str | None = None  # gRPC ssl_target_name_override, for certs not matching the host

    TOKEN: str | None = None  # sent as Authorization: Bearer <TOKEN> when set
    EXTRA_HEADERS: dict[str, str] = Field(default_factory=dict)

    CHAT_TEMPLATE_MODEL: str | None = None  # HF tokenizer id used by chat/templates.py

    LOG_LEVEL: str = "INFO"


@lru_cache
def get_settings() -> TritonSettings:
    """Build (and cache) the process-wide :class:`TritonSettings` from the environment.

    :return: A memoized :class:`TritonSettings` instance; the same object is returned on every call.
    """
    return TritonSettings()
