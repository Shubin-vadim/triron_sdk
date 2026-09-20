"""Shared wiring: settings -> transport, model config cache, backend resolution."""

from __future__ import annotations

import logging

from ..backends import BackendAdapter, detect_backend, get_adapter
from ..chat import ChatTemplate, SimpleChatTemplate
from ..core.constants import BackendKind, Protocol
from ..core.models import ModelConfig
from ..settings import TritonSettings, get_settings
from ..transport import TransportInterface, create_transport
from ..transport.tls import TLSConfig


logger = logging.getLogger(__name__)


class BaseClient:
    """Shared plumbing for the sync and async Triton clients: transport, caches, adapter resolution."""

    def __init__(
        self,
        url: str | None = None,
        *,
        protocol: Protocol | str | None = None,
        backend: BackendKind | str | None = None,
        settings: TritonSettings | None = None,
        transport: TransportInterface | None = None,
        chat_template: ChatTemplate | None = None,
    ) -> None:
        """Build the transport and per-model caches.

        :param url: Server address; the settings' ``URL`` when omitted.
        :param protocol: Wire protocol to use; the settings' ``PROTOCOL`` when omitted.
        :param backend: Force a specific backend adapter instead of auto-detecting it from the
            model config.
        :param settings: Settings instance to read defaults from; loaded from the environment
            when omitted.
        :param transport: A pre-built transport to use instead of constructing one (for tests).
        :param chat_template: Template used to render chat messages; :class:`SimpleChatTemplate`
            when omitted.
        :raises ValueError: If ``protocol`` resolves to ``"openai"`` (no tensor transport exists
            for it; use :class:`triton_sdk.openai.OpenAICompatClient` instead).
        """
        self._settings = settings or get_settings()
        self._url = url or self._settings.URL
        self._protocol = Protocol(protocol or self._settings.PROTOCOL)
        self._forced_backend = BackendKind(backend) if backend else self._settings.BACKEND
        self._transport = transport or self._build_transport()
        self._chat_template = chat_template or SimpleChatTemplate()
        self._config_cache: dict[str, ModelConfig] = {}
        self._adapter_cache: dict[str, BackendAdapter] = {}

    # --- wiring ------------------------------------------------------------
    def _build_transport(self) -> TransportInterface:
        """Construct the transport implied by :attr:`_protocol` and the bound settings.

        :return: The constructed :class:`~triton_sdk.transport.base.TransportInterface`.
        :raises ValueError: If :attr:`_protocol` is ``"openai"``.
        """
        settings = self._settings
        headers = dict(settings.EXTRA_HEADERS)
        if settings.TOKEN:
            headers["Authorization"] = f"Bearer {settings.TOKEN}"
        if self._protocol is Protocol.OPENAI:
            msg = "protocol=openai has no tensor transport; use triton_sdk.openai.OpenAICompatClient"
            raise ValueError(msg)
        tls = TLSConfig.from_settings(settings)
        if self._protocol is Protocol.GRPC:
            return create_transport("grpc", self._url, timeout=settings.TIMEOUT, metadata=headers, tls=tls)
        return create_transport(
            "http",
            self._url,
            timeout=settings.TIMEOUT,
            connect_timeout=settings.CONNECT_TIMEOUT,
            headers=headers,
            tls=tls,
        )

    async def _resolve_adapter(self, model: str, version: str | None = None) -> BackendAdapter:
        """Pick the backend adapter for a model, once, then reuse it.

        An explicit ``BACKEND`` setting wins over auto-detection; otherwise the adapter is chosen
        by inspecting the model's config.

        :param model: Model name.
        :param version: Model version to fetch the config for, if the config is not cached yet.
        :return: The resolved :class:`~triton_sdk.backends.base.BackendAdapter`.
        """
        if model in self._adapter_cache:
            return self._adapter_cache[model]
        config = await self._get_config(model, version)
        if self._forced_backend:
            adapter = get_adapter(self._forced_backend, config)
        else:
            adapter = detect_backend(config)
            logger.debug("model %s -> backend %s", model, adapter.kind)
        self._adapter_cache[model] = adapter
        return adapter

    async def _get_config(self, model: str, version: str | None = None) -> ModelConfig:
        """Fetch a model's config, caching it by model name.

        :param model: Model name.
        :param version: Model version; the server's default version when omitted.
        :return: The (possibly cached) :class:`~triton_sdk.core.models.ModelConfig`.
        """
        if model not in self._config_cache:
            self._config_cache[model] = await self._transport.model_config(model, version)
        return self._config_cache[model]

    def resolve_model(self, model: str | None) -> str:
        """Resolve the model name to use for a call, falling back to the configured default.

        :param model: Model name passed by the caller, or ``None`` to use the default.
        :return: The resolved model name.
        :raises ValueError: If ``model`` is ``None`` and no default model is configured.
        """
        resolved = model or self._settings.DEFAULT_MODEL
        if not resolved:
            msg = "model is required: pass model=... or set TRITON_DEFAULT_MODEL"
            raise ValueError(msg)
        return resolved

    @property
    def transport(self) -> TransportInterface:
        """The underlying transport.

        :return: The :class:`~triton_sdk.transport.base.TransportInterface` this client uses.
        """
        return self._transport

    @property
    def chat_template(self) -> ChatTemplate:
        """The chat template used to render messages.

        :return: The bound :class:`~triton_sdk.chat.ChatTemplate`.
        """
        return self._chat_template
