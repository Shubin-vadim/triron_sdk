"""Adapter for the vLLM backend (triton-inference-server/vllm_backend)."""

from __future__ import annotations

import json

from ..core.constants import FINAL_RESPONSE_PARAM, TEXT_INPUT, TEXT_OUTPUT, BackendKind
from ..core.models import DataType, GenerateChunk, GenerateRequest, InferInput, InferResult, ModelConfig
from ..core.utils import drop_none
from .base import BackendAdapter


class VLLMAdapter(BackendAdapter):
    """Adapter for Triton's vllm_backend: one JSON-encoded sampling_parameters tensor."""

    kind = BackendKind.VLLM
    default_batched = False  # vllm_backend models declare max_batch_size: 0 -> shapes are [N]
    #: vllm_backend streams text deltas (verified on Triton 25.08). Very old releases sent the cumulative text;
    #: set this to True for those deployments and the adapter will subtract the text seen so far.
    cumulative_text = False

    @classmethod
    def matches(cls, config: ModelConfig) -> bool:
        """Match models declaring a ``sampling_parameters`` input, or backend name "vllm".

        :param config: Model config to inspect.
        :return: ``True`` if the config looks like a vllm_backend deployment.
        """
        return "sampling_parameters" in config.input_names or config.backend == "vllm"

    def build_inputs(self, request: GenerateRequest) -> list[InferInput]:
        """Build vllm_backend's tensors: text_input, stream, sampling_parameters (JSON).

        :param request: The request to translate.
        :return: The input tensors to send to Triton.
        """
        params = request.params
        sampling = drop_none(
            {
                "max_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "repetition_penalty": params.repetition_penalty,
                "presence_penalty": params.presence_penalty,
                "frequency_penalty": params.frequency_penalty,
                "min_tokens": params.min_tokens,
                "seed": params.seed,
                "stop": params.stop or None,
                "ignore_eos": params.ignore_eos or None,
            }
        )
        inputs = [
            self.tensor(TEXT_INPUT, [request.prompt], DataType.BYTES),
            self.tensor("stream", [request.stream], DataType.BOOL),
            self.tensor("sampling_parameters", [json.dumps(sampling)], DataType.BYTES),
        ]
        exclude = self.optional_tensor("exclude_input_in_output", request.exclude_input_in_output, DataType.BOOL)
        if exclude is not None:
            inputs.append(exclude)
        inputs.extend(self.extra_inputs(request))
        return inputs

    def requested_outputs(self, request: GenerateRequest) -> list[str]:
        """Request only the ``text_output`` tensor.

        :param request: The request being served; unused, kept for interface compatibility.
        :return: A single-element list containing ``text_output``.
        """
        return [TEXT_OUTPUT]

    def parse_result(self, result: InferResult, *, index: int, previous_text: str) -> GenerateChunk:
        """Parse a vllm_backend response into a chunk.

        By default the text is already a delta; when :attr:`cumulative_text` is set (for older
        backend releases), the text seen so far is subtracted.

        :param result: The raw inference result to parse.
        :param index: Zero-based position of this chunk within the stream.
        :param previous_text: Text already emitted by earlier chunks.
        :return: The parsed :class:`~triton_sdk.core.models.GenerateChunk`.
        """
        text = str(result.output(TEXT_OUTPUT).first() or "") if TEXT_OUTPUT in result.outputs else ""
        if self.cumulative_text and text.startswith(previous_text):
            text = text[len(previous_text) :]
        is_final = bool(result.parameters.get(FINAL_RESPONSE_PARAM, False))
        return GenerateChunk(text=text, index=index, is_final=is_final, raw=result.model_dump())
