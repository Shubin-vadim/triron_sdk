"""Backend-agnostic request models. Backend adapters translate these into Triton tensors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SamplingParams(BaseModel):
    """Superset of TensorRT-LLM and vLLM sampling knobs. Unsupported fields are dropped by the adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tokens: int = Field(default=256, ge=1)
    temperature: float | None = Field(default=None, ge=0.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    length_penalty: float | None = None
    min_tokens: int | None = Field(default=None, ge=0)
    beam_width: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    bad_words: list[str] = Field(default_factory=list)
    return_log_probs: bool = False
    ignore_eos: bool = False


class GenerateRequest(BaseModel):
    """One prompt -> one completion (or one stream of chunks)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Model name as it appears in the Triton model repository.
    model: str
    #: Raw prompt text (already through any chat template).
    prompt: str
    params: SamplingParams = Field(default_factory=SamplingParams)
    stream: bool = False
    model_version: str | None = None
    request_id: str | None = None
    exclude_input_in_output: bool = True
    extra_inputs: dict[str, object] = Field(default_factory=dict)  # escape hatch for custom ensembles
