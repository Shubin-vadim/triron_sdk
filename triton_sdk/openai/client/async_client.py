"""Async openai-shaped client: picks an engine and exposes the resource namespaces."""

from __future__ import annotations

from typing import Self

from ...chat import ChatTemplate, HFChatTemplate
from ...client import AsyncTritonLLMClient
from ...core.constants import BackendKind, Protocol
from ...settings import TritonSettings, get_settings
from ...transport import TransportInterface
from ...transport.tls import TLSConfig
from ..engines import Engine, FrontendEngine, NativeEngine
from ..frontend import AsyncOpenAIFrontend
from ..resources import AsyncChat, AsyncCompletions, AsyncModels


def _resolve_template(tokenizer: str | None, chat_template: ChatTemplate | None) -> ChatTemplate | None:
    """Pick the chat template to use: an explicit instance, or one built from a tokenizer id.

    :param tokenizer: Hugging Face tokenizer id to build an :class:`HFChatTemplate` from.
    :param chat_template: An already-built template, which takes precedence over ``tokenizer``.
    :return: The template to use, or ``None`` when neither argument is given (the native client
        then falls back to its own default).
    """
    if chat_template is not None:
        return chat_template
    return HFChatTemplate(tokenizer) if tokenizer else None


class AsyncOpenAICompatClient:
    """Async, openai-shaped client for Triton, over either KServe v2 or the OpenAI frontend."""

    def __init__(
        self,
        url: str | None = None,
        *,
        protocol: Protocol | str | None = None,
        backend: BackendKind | str | None = None,
        tokenizer: str | None = None,
        chat_template: ChatTemplate | None = None,
        settings: TritonSettings | None = None,
        transport: TransportInterface | None = None,
    ) -> None:
        """Build the engine implied by ``protocol`` and wire up the resource namespaces.

        :param url: Server address; the settings' ``URL`` when omitted.
        :param protocol: ``"http"`` / ``"grpc"`` for the native engine, or ``"openai"`` for
            Triton's OpenAI frontend; the settings' ``PROTOCOL`` when omitted.
        :param backend: Force a specific backend adapter (native engine only).
        :param tokenizer: Hugging Face tokenizer id used to build a chat template and compute
            usage (native engine only).
        :param chat_template: An already-built chat template, taking precedence over
            ``tokenizer`` (native engine only).
        :param settings: Settings instance to read defaults from; loaded from the environment
            when omitted.
        :param transport: A pre-built transport for the native engine (for tests).
        """
        settings = settings or get_settings()
        resolved_protocol = Protocol(protocol or settings.PROTOCOL)
        self._native: AsyncTritonLLMClient | None = None
        self._engine: Engine
        if resolved_protocol is Protocol.OPENAI:
            headers = dict(settings.EXTRA_HEADERS)
            if settings.TOKEN:
                headers["Authorization"] = f"Bearer {settings.TOKEN}"
            frontend = AsyncOpenAIFrontend(
                url or settings.URL,
                timeout=settings.TIMEOUT,
                stream_timeout=settings.STREAM_TIMEOUT,
                connect_timeout=settings.CONNECT_TIMEOUT,
                headers=headers,
                tls=TLSConfig.from_settings(settings),
            )
            self._engine = FrontendEngine(frontend, default_model=settings.DEFAULT_MODEL)
        else:
            self._native = AsyncTritonLLMClient(
                url,
                protocol=resolved_protocol,
                backend=backend,
                settings=settings,
                transport=transport,
                chat_template=_resolve_template(tokenizer, chat_template),
            )
            self._engine = NativeEngine(self._native)

        self.chat = AsyncChat(self._engine)
        self.completions = AsyncCompletions(self._engine)
        self.models = AsyncModels(self._engine)

    @property
    def native(self) -> AsyncTritonLLMClient | None:
        """The underlying Triton client.

        :return: The wrapped :class:`~triton_sdk.client.async_client.AsyncTritonLLMClient`, or
            ``None`` when talking to the OpenAI frontend.
        """
        return self._native

    async def is_ready(self) -> bool:
        """Check server/frontend readiness.

        :return: ``True`` if ready to serve requests.
        """
        return await self._engine.ready()

    async def __aenter__(self) -> Self:
        """Enter the async context manager.

        :return: This client instance.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager, closing the client.

        :param exc_info: Standard ``(exc_type, exc_value, traceback)`` tuple, unused.
        """
        await self.close()

    async def close(self) -> None:
        """Close the underlying engine (client or frontend connection)."""
        await self._engine.close()
