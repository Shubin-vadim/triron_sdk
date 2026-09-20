"""State and bookkeeping shared by the chat and completion stream assemblers."""

from __future__ import annotations

from ...chat import ChatTemplate
from ...core.models import FinishReason as SdkFinishReason
from ...core.models import GenerateChunk, GenerateResponse
from ..convert import build_usage, finish_reason
from ..types import CompletionUsage, FinishReason, new_id, now


class StreamAssemblerBase:
    """Tracks one stream's identity (id, model, created) and the text emitted so far."""

    id_prefix: str = "chatcmpl"

    def __init__(
        self, *, model: str, prompt: str, template: ChatTemplate, max_tokens: int, include_usage: bool
    ) -> None:
        """Initialize the assembler for one stream.

        :param model: Model name to attach to every emitted chunk.
        :param prompt: The rendered prompt text, used to compute usage at the end of the stream.
        :param template: Chat template used to compute usage.
        :param max_tokens: The token budget that was requested, used to infer the finish reason.
        :param include_usage: Whether to emit a trailing usage-only chunk in ``finish()``.
        """
        self._model = model
        self._prompt = prompt
        self._template = template
        self._max_tokens = max_tokens
        self._include_usage = include_usage
        self._id = new_id(self.id_prefix)
        self._created = now()
        self._parts: list[str] = []
        self._last: GenerateChunk | None = None

    def _record(self, chunk: GenerateChunk) -> bool:
        """Remember a chunk and its text.

        :param chunk: The SDK chunk just received.
        :return: ``True`` if the chunk carried text (and should be emitted), else ``False``.
        """
        self._last = chunk
        if not chunk.text:
            return False
        self._parts.append(chunk.text)
        return True

    def _summary(self) -> tuple[CompletionUsage | None, FinishReason]:
        """Compute usage and finish reason for the stream seen so far.

        :return: A ``(usage, finish_reason)`` tuple; ``usage`` is ``None`` when the template has
            no tokenizer.
        """
        completion = "".join(self._parts)
        usage = build_usage(self._template, self._prompt, completion)
        summary = GenerateResponse(
            model=self._model,
            text=completion,
            finish_reason=(
                self._last.finish_reason if self._last and self._last.finish_reason else SdkFinishReason.UNKNOWN
            ),
        )
        return usage, finish_reason(summary, usage=usage, max_tokens=self._max_tokens)
