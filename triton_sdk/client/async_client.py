"""Async public API. All LLM semantics (chunks -> response, chat -> prompt) live here."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any, Self

from ..chat import ChatMessage
from ..core.models import (
    GenerateChunk,
    GenerateRequest,
    GenerateResponse,
    ModelConfig,
    ModelMetadata,
    RepositoryModel,
    SamplingParams,
)
from ..core.utils import build_retrying
from ..exceptions import ModelNotReadyError
from .base import BaseClient


logger = logging.getLogger(__name__)


class AsyncTritonLLMClient(BaseClient):
    """Async client for Triton LLM models: readiness, metadata, generation and chat."""

    # --- lifecycle ---------------------------------------------------------
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
        """Close the underlying transport. Safe to call more than once."""
        await self._transport.close()

    # --- health / metadata -------------------------------------------------
    async def is_ready(self, model: str | None = None) -> bool:
        """Check server or model readiness.

        :param model: Model name to check; when omitted, checks overall server readiness instead.
        :return: ``True`` if the server (or model) is ready.
        """
        if model is None:
            return await self._transport.server_ready()
        return await self._transport.model_ready(self.resolve_model(model))

    async def wait_for_ready(self, model: str | None = None, *, timeout: float = 300.0, interval: float = 2.0) -> None:
        """Poll readiness until the server (or model) is ready.

        :param model: Model name to wait for; when omitted, waits for overall server readiness.
        :param timeout: Maximum time to wait, in seconds.
        :param interval: Delay between readiness checks, in seconds.
        :raises ModelNotReadyError: If the deadline passes without the target becoming ready.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            try:
                if await self.is_ready(model):
                    return
            except Exception as exc:  # noqa: BLE001 - server may not accept connections yet
                logger.debug("readiness probe failed: %s", exc)
            await asyncio.sleep(interval)
        target = model or "server"
        msg = f"{target} not ready after {timeout}s"
        raise ModelNotReadyError(msg, model=model)

    async def model_metadata(self, model: str | None = None) -> ModelMetadata:
        """Fetch a model's declared inputs, outputs and available versions.

        :param model: Model name; the configured default model when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelMetadata`.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        """
        return await self._transport.model_metadata(self.resolve_model(model))

    async def model_config(self, model: str | None = None) -> ModelConfig:
        """Fetch (and cache) a model's config.pbtxt.

        :param model: Model name; the configured default model when omitted.
        :return: The parsed :class:`~triton_sdk.core.models.ModelConfig`.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        """
        return await self._get_config(self.resolve_model(model))

    async def list_models(self) -> list[RepositoryModel]:
        """List every model the server repository knows about, with its load state.

        :return: One :class:`~triton_sdk.core.models.RepositoryModel` per model/version.
        """
        return await self._transport.repository_index()

    # --- generation --------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        *,
        model: str | None = None,
        model_version: str | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
        extra_inputs: dict[str, Any] | None = None,
        exclude_input_in_output: bool = True,
    ) -> GenerateResponse:
        """Generate one completion for a prompt.

        Retries on transient failures per the client's retry settings. For decoupled models this
        transparently uses the streaming API and collects the chunks, since Triton rejects unary
        inference for them.

        :param prompt: Raw prompt text.
        :param params: Sampling parameters; :class:`~triton_sdk.core.models.SamplingParams`
            defaults when omitted.
        :param model: Model name; the configured default model when omitted.
        :param model_version: Model version; the server's default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param timeout: Per-call timeout, in seconds; the configured default when omitted.
        :param extra_inputs: Additional model inputs for custom ensembles (plain values or
            :class:`~triton_sdk.core.models.InferInput`); datatypes come from the model config.
        :param exclude_input_in_output: When ``False``, asks the backend to echo the prompt back
            (backend permitting).
        :return: The assembled :class:`~triton_sdk.core.models.GenerateResponse`.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        :raises triton_sdk.exceptions.TritonError: If generation fails after all retries.
        """
        request = GenerateRequest(
            model=self.resolve_model(model),
            prompt=prompt,
            params=params or SamplingParams(),
            model_version=model_version,
            request_id=request_id,
            exclude_input_in_output=exclude_input_in_output,
            extra_inputs=extra_inputs or {},
        )
        retrying = build_retrying(self._settings.RETRY_COUNT, self._settings.RETRY_START_TIMEOUT)
        async for attempt in retrying:
            with attempt:
                return await self._generate_once(request, timeout=timeout)
        msg = "unreachable: tenacity reraises the last error"
        raise AssertionError(msg)

    async def _generate_once(self, request: GenerateRequest, *, timeout: float | None) -> GenerateResponse:
        """Run a single (non-retried) attempt of :meth:`generate`.

        :param request: The request to send.
        :param timeout: Per-call timeout, in seconds; the configured default when omitted.
        :return: The assembled :class:`~triton_sdk.core.models.GenerateResponse`.
        :raises triton_sdk.exceptions.TritonError: On any transport or inference failure.
        """
        adapter = await self._resolve_adapter(request.model, request.model_version)
        call: dict[str, Any] = {
            "outputs": adapter.requested_outputs(request),
            "version": request.model_version,
            "request_id": request.request_id,
            "timeout": timeout or self._settings.TIMEOUT,
        }
        if adapter.supports_streaming():
            # Triton rejects unary ModelInfer / POST infer for decoupled models ("doesn't support models with
            # decoupled transaction policy"), so a non-streaming request still travels over the stream API with
            # stream=False and is collected here. Verified against Triton 25.08 + vllm_backend.
            chunks: list[GenerateChunk] = []
            previous_text = ""
            async with aclosing(
                self._transport.stream_infer(request.model, adapter.build_inputs(request), **call)
            ) as s:
                async for result in s:
                    chunk = adapter.parse_result(result, index=len(chunks), previous_text=previous_text)
                    previous_text += chunk.text
                    chunks.append(chunk)
            return GenerateResponse.from_chunks(request.model, chunks, request_id=request.request_id)

        result = await self._transport.infer(request.model, adapter.build_inputs(request), **call)
        chunk = adapter.parse_result(result, index=0, previous_text="")
        return GenerateResponse.from_chunks(request.model, [chunk], request_id=request.request_id)

    async def stream(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        *,
        model: str | None = None,
        model_version: str | None = None,
        request_id: str | None = None,
        timeout: float | None = None,
        extra_inputs: dict[str, Any] | None = None,
        exclude_input_in_output: bool = True,
    ) -> AsyncIterator[GenerateChunk]:
        """Stream a completion for a prompt, chunk by chunk.

        Breaking out of the ``async for`` loop closes the underlying stream, which cancels the
        request on the server (over gRPC) or drops the connection (over HTTP).

        :param prompt: Raw prompt text.
        :param params: Sampling parameters; :class:`~triton_sdk.core.models.SamplingParams`
            defaults when omitted.
        :param model: Model name; the configured default model when omitted.
        :param model_version: Model version; the server's default version when omitted.
        :param request_id: Caller-supplied id to correlate this request in logs and traces.
        :param timeout: Maximum idle time between two chunks, in seconds; the configured
            ``STREAM_TIMEOUT`` when omitted.
        :param extra_inputs: Additional model inputs for custom ensembles (plain values or
            :class:`~triton_sdk.core.models.InferInput`); datatypes come from the model config.
        :param exclude_input_in_output: When ``False``, asks the backend to echo the prompt back
            (backend permitting).
        :return: An async iterator of :class:`~triton_sdk.core.models.GenerateChunk`.
        :raises ValueError: If ``model`` is omitted and no default model is configured.
        :raises triton_sdk.exceptions.TritonError: On any transport or inference failure.
        """
        request = GenerateRequest(
            model=self.resolve_model(model),
            prompt=prompt,
            params=params or SamplingParams(),
            stream=True,
            model_version=model_version,
            request_id=request_id,
            exclude_input_in_output=exclude_input_in_output,
            extra_inputs=extra_inputs or {},
        )
        adapter = await self._resolve_adapter(request.model, request.model_version)
        previous_text = ""
        index = 0
        results = self._transport.stream_infer(
            request.model,
            adapter.build_inputs(request),
            outputs=adapter.requested_outputs(request),
            version=request.model_version,
            request_id=request.request_id,
            timeout=timeout or self._settings.STREAM_TIMEOUT,
        )
        # aclosing() guarantees the transport generator is closed (and the request cancelled) as soon as the
        # caller breaks out of the loop, instead of whenever the garbage collector gets to it.
        async with aclosing(results) as stream:
            async for result in stream:
                chunk = adapter.parse_result(result, index=index, previous_text=previous_text)
                previous_text += chunk.text
                index += 1
                if chunk.text or chunk.is_final:
                    yield chunk
                if chunk.is_final:
                    break

    # --- chat --------------------------------------------------------------
    def render_chat(self, messages: list[ChatMessage]) -> str:
        """Render a conversation into a raw prompt using the bound chat template.

        :param messages: The conversation, in order.
        :return: The rendered prompt string.
        """
        return self._chat_template.render(messages)

    async def chat(
        self, messages: list[ChatMessage], params: SamplingParams | None = None, **kwargs: object
    ) -> GenerateResponse:
        """Render a conversation and generate a completion for it.

        :param messages: The conversation, in order.
        :param params: Sampling parameters; see :meth:`generate`.
        :param kwargs: Extra keyword arguments forwarded to :meth:`generate`.
        :return: The assembled :class:`~triton_sdk.core.models.GenerateResponse`.
        """
        return await self.generate(self.render_chat(messages), params, **kwargs)  # type: ignore[arg-type]

    def chat_stream(
        self, messages: list[ChatMessage], params: SamplingParams | None = None, **kwargs: object
    ) -> AsyncIterator[GenerateChunk]:
        """Render a conversation and stream a completion for it.

        :param messages: The conversation, in order.
        :param params: Sampling parameters; see :meth:`stream`.
        :param kwargs: Extra keyword arguments forwarded to :meth:`stream`.
        :return: An async iterator of :class:`~triton_sdk.core.models.GenerateChunk`.
        """
        return self.stream(self.render_chat(messages), params, **kwargs)  # type: ignore[arg-type]
