"""Assembles Completion chunks for a streamed legacy (non-chat) completion.

Like openai, every streamed chunk is a full Completion object whose ``choices[0].text`` holds the delta; the last
one carries ``finish_reason`` and empty text, optionally followed by a usage-only chunk.
"""

from __future__ import annotations

from ...core.models import GenerateChunk
from ..types import Completion, CompletionChoice
from .base import StreamAssemblerBase


class CompletionStreamAssembler(StreamAssemblerBase):
    """Legacy completions stream: every chunk is a Completion whose choice.text holds the delta."""

    id_prefix = "cmpl"

    def _chunk(self, choices: list[CompletionChoice], usage: object = None) -> Completion:
        """Build one Completion sharing this stream's id, model and creation time.

        :param choices: Choices to attach to the chunk.
        :param usage: Usage to attach to the chunk, if any.
        :return: The constructed :class:`~triton_sdk.openai.types.Completion`.
        """
        return Completion(id=self._id, created=self._created, model=self._model, choices=choices, usage=usage)  # type: ignore[arg-type]

    def delta(self, chunk: GenerateChunk) -> Completion | None:
        """Convert one SDK generation chunk into a Completion, if it carries text.

        :param chunk: The SDK chunk to convert; also recorded as the last chunk seen for
            :meth:`finish`.
        :return: The converted chunk, or ``None`` when ``chunk.text`` is empty.
        """
        if not self._record(chunk):
            return None
        return self._chunk([CompletionChoice(text=chunk.text)])

    def finish(self) -> list[Completion]:
        """Build the closing chunk(s): a finish_reason chunk, plus a usage chunk when requested.

        :return: One chunk with empty text and ``finish_reason`` set, followed by a usage-only
            chunk when ``include_usage`` was set at construction.
        """
        usage, reason = self._summary()
        chunks = [self._chunk([CompletionChoice(text="", finish_reason=reason)])]
        if self._include_usage:
            chunks.append(self._chunk([], usage=usage))
        return chunks
