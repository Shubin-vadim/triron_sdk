"""Assembles ChatCompletionChunk objects for a streamed chat completion.

Mirrors the wire behaviour of the OpenAI API: a first chunk carrying only role="assistant", one chunk per delta,
a final chunk with finish_reason and an empty delta, and (with stream_options.include_usage) a trailing chunk with
empty choices and usage.
"""

from __future__ import annotations

from ...core.models import GenerateChunk
from ..types import ChatCompletionChunk, ChoiceDelta, ChunkChoice
from .base import StreamAssemblerBase


class ChatStreamAssembler(StreamAssemblerBase):
    """Builds the openai-style ChatCompletionChunk sequence for one streamed chat completion."""

    id_prefix = "chatcmpl"

    def _chunk(self, choices: list[ChunkChoice], usage: object = None) -> ChatCompletionChunk:
        """Build one ChatCompletionChunk sharing this stream's id, model and creation time.

        :param choices: Choices to attach to the chunk.
        :param usage: Usage to attach to the chunk, if any.
        :return: The constructed :class:`~triton_sdk.openai.types.ChatCompletionChunk`.
        """
        return ChatCompletionChunk(id=self._id, created=self._created, model=self._model, choices=choices, usage=usage)  # type: ignore[arg-type]

    def start(self) -> ChatCompletionChunk:
        """Build the first chunk of the stream, carrying only ``role="assistant"``.

        :return: The initial :class:`~triton_sdk.openai.types.ChatCompletionChunk`.
        """
        return self._chunk([ChunkChoice(delta=ChoiceDelta(role="assistant", content=""))])

    def delta(self, chunk: GenerateChunk) -> ChatCompletionChunk | None:
        """Convert one SDK generation chunk into a ChatCompletionChunk, if it carries text.

        :param chunk: The SDK chunk to convert; also recorded as the last chunk seen for
            :meth:`finish`.
        :return: The converted chunk, or ``None`` when ``chunk.text`` is empty.
        """
        if not self._record(chunk):
            return None
        return self._chunk([ChunkChoice(delta=ChoiceDelta(content=chunk.text))])

    def finish(self) -> list[ChatCompletionChunk]:
        """Build the closing chunk(s): a finish_reason chunk, plus a usage chunk when requested.

        :return: One chunk with an empty delta and ``finish_reason`` set, followed by a
            usage-only chunk when ``include_usage`` was set at construction.
        """
        usage, reason = self._summary()
        chunks = [self._chunk([ChunkChoice(delta=ChoiceDelta(), finish_reason=reason)])]
        if self._include_usage:
            chunks.append(self._chunk([], usage=usage))
        return chunks
