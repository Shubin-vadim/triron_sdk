"""Backend-agnostic response models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FinishReason(StrEnum):
    """Why generation stopped. Triton backends rarely report this explicitly."""

    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"
    ERROR = "error"
    UNKNOWN = "unknown"


class Usage(BaseModel):
    """Prompt/completion token counts, when the caller supplied a tokenizer to compute them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def total_tokens(self) -> int | None:
        """Sum of prompt and completion tokens.

        :return: ``prompt_tokens + completion_tokens``, or ``None`` if either count is unknown.
        """
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


class GenerateChunk(BaseModel):
    """One streamed piece of text. Backends that stream cumulative text are normalised to deltas."""

    model_config = ConfigDict(frozen=True)

    text: str
    index: int = 0
    is_final: bool = False
    finish_reason: FinishReason | None = None
    log_probs: list[float] | None = None
    cum_log_probs: float | None = None
    raw: dict[str, object] = Field(default_factory=dict)  # untouched backend payload for debugging


class GenerateResponse(BaseModel):
    """A completion assembled from one or more :class:`GenerateChunk` (unary call or collected stream)."""

    model_config = ConfigDict(frozen=True)

    model: str
    text: str
    finish_reason: FinishReason = FinishReason.UNKNOWN
    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    cum_log_probs: float | None = None
    raw: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_chunks(cls, model: str, chunks: list[GenerateChunk], request_id: str | None = None) -> GenerateResponse:
        """Concatenate streamed chunks into a single response.

        :param model: Model name to record on the response.
        :param chunks: Chunks in stream order; their :attr:`~GenerateChunk.text` is joined and the last
            chunk's :attr:`~GenerateChunk.finish_reason` is used.
        :param request_id: Request id to record on the response, if any.
        :return: The assembled :class:`GenerateResponse`.
        """
        last = chunks[-1] if chunks else None
        cum_log_probs = next((c.cum_log_probs for c in reversed(chunks) if c.cum_log_probs is not None), None)
        return cls(
            model=model,
            text="".join(chunk.text for chunk in chunks),
            finish_reason=(last.finish_reason if last and last.finish_reason else FinishReason.UNKNOWN),
            request_id=request_id,
            cum_log_probs=cum_log_probs,
        )
