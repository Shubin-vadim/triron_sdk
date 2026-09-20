"""Fallback adapter: any model with text_input/text_output, driven through the /generate endpoints."""

from __future__ import annotations

from ..core.constants import TEXT_INPUT, TEXT_OUTPUT, BackendKind
from ..core.models import DataType, GenerateChunk, GenerateRequest, InferInput, InferResult, ModelConfig
from .base import BackendAdapter


class GenericAdapter(BackendAdapter):
    """Fallback adapter for any model exposing plain text_input/text_output tensors."""

    kind = BackendKind.GENERIC
    default_batched = False

    @classmethod
    def matches(cls, config: ModelConfig) -> bool:
        """Match any model that declares a ``text_input`` input.

        :param config: Model config to inspect.
        :return: ``True`` if ``text_input`` is among the declared inputs.
        """
        return TEXT_INPUT in config.input_names

    def build_inputs(self, request: GenerateRequest) -> list[InferInput]:
        """Build the ``text_input`` tensor plus any escape-hatch extra inputs.

        :param request: The request to translate.
        :return: The input tensors to send to Triton.
        """
        inputs = [self.tensor(TEXT_INPUT, [request.prompt], DataType.BYTES)]
        inputs.extend(self.extra_inputs(request))
        return inputs

    def requested_outputs(self, request: GenerateRequest) -> list[str]:
        """Request only the ``text_output`` tensor.

        :param request: The request being served; unused, kept for interface compatibility.
        :return: A single-element list containing ``text_output``.
        """
        return [TEXT_OUTPUT]

    def parse_result(self, result: InferResult, *, index: int, previous_text: str) -> GenerateChunk:
        """Parse the ``text_output`` tensor as a single, final chunk.

        :param result: The raw inference result to parse.
        :param index: Zero-based position of this chunk within the stream.
        :param previous_text: Unused; this backend does not stream.
        :return: A :class:`~triton_sdk.core.models.GenerateChunk` with ``is_final=True``.
        """
        text = str(result.output(TEXT_OUTPUT).first() or "")
        return GenerateChunk(text=text, index=index, is_final=True, raw=result.model_dump())
