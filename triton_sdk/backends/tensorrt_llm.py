"""Adapter for the TensorRT-LLM backend: the `ensemble` (preprocessing -> tensorrt_llm -> postprocessing) and the
`tensorrt_llm_bls` model from tensorrtllm_backend/all_models/inflight_batcher_llm.

Both expose the same text-level schema: one tensor per sampling knob, BYTES tensors for words lists, `text_output`
(+ optional log-prob tensors) on the way back. Tensor names drifted between backend releases (min_length ->
min_tokens), so every optional knob is resolved against the model config when one is available.
"""

from __future__ import annotations

import logging

from ..core.constants import FINAL_RESPONSE_PARAM, TEXT_INPUT, TEXT_OUTPUT, BackendKind
from ..core.models import DataType, GenerateChunk, GenerateRequest, InferInput, InferResult, ModelConfig
from .base import BackendAdapter


logger = logging.getLogger(__name__)

# SamplingParams field -> (candidate tensor names, datatype). Order of candidates = newest name first.
_SCALAR_KNOBS: tuple[tuple[str, tuple[str, ...], DataType], ...] = (
    ("temperature", ("temperature",), DataType.FP32),
    ("top_p", ("top_p",), DataType.FP32),
    ("top_k", ("top_k",), DataType.INT32),
    ("repetition_penalty", ("repetition_penalty",), DataType.FP32),
    ("presence_penalty", ("presence_penalty",), DataType.FP32),
    ("frequency_penalty", ("frequency_penalty",), DataType.FP32),
    ("length_penalty", ("length_penalty",), DataType.FP32),
    ("min_tokens", ("min_tokens", "min_length"), DataType.INT32),
    ("beam_width", ("beam_width",), DataType.INT32),
    ("seed", ("random_seed",), DataType.UINT64),
)

_LOG_PROB_OUTPUTS: tuple[str, ...] = ("cum_log_probs", "output_log_probs")

# TensorRT-LLM has no ignore_eos flag. The executor treats end_id == -1 as "no end token", so generation runs
# until max_tokens; that is what vLLM does for ignore_eos=True.
IGNORE_EOS_END_ID = -1


class TensorRTLLMAdapter(BackendAdapter):
    """Adapter for the TensorRT-LLM ensemble / tensorrt_llm_bls model: one tensor per sampling knob."""

    kind = BackendKind.TENSORRT_LLM
    default_batched = True  # ensemble / BLS configs set max_batch_size > 0
    known_input_types = {
        "end_id": DataType.INT32,
        "pad_id": DataType.INT32,
        "embedding_bias_words": DataType.BYTES,
        "embedding_bias_weights": DataType.FP32,
        "return_context_logits": DataType.BOOL,
        "return_generation_logits": DataType.BOOL,
        "return_num_input_tokens": DataType.BOOL,
        "return_num_output_tokens": DataType.BOOL,
        "prompt_vocab_size": DataType.INT32,
        "num_draft_tokens": DataType.INT32,
        "use_draft_logits": DataType.BOOL,
        "lora_task_id": DataType.UINT64,
        "guided_decoding_guide_type": DataType.BYTES,
        "guided_decoding_guide": DataType.BYTES,
    }

    @classmethod
    def matches(cls, config: ModelConfig) -> bool:
        """Match models declaring text_input + max_tokens but no sampling_parameters (vLLM's marker).

        :param config: Model config to inspect.
        :return: ``True`` if the config looks like a TensorRT-LLM ensemble/BLS deployment.
        """
        names = config.input_names
        return TEXT_INPUT in names and "max_tokens" in names and "sampling_parameters" not in names

    # --- request ---------------------------------------------------------------
    def build_inputs(self, request: GenerateRequest) -> list[InferInput]:
        """Build one tensor per declared sampling knob, plus words lists and boolean flags.

        :param request: The request to translate.
        :return: The input tensors to send to Triton.
        """
        params = request.params
        inputs: list[InferInput] = [
            self.tensor(TEXT_INPUT, [request.prompt], DataType.BYTES),
            self.tensor("max_tokens", [params.max_tokens], DataType.INT32),
        ]
        self._append(inputs, self.optional_tensor("stream", request.stream, DataType.BOOL))

        for field, candidates, datatype in _SCALAR_KNOBS:
            value = getattr(params, field)
            if value is None:
                continue
            name = self.resolve_input(*candidates)
            if name is None:
                logger.debug("model %s does not accept %s (%s); dropping", self._model_name, field, candidates)
                continue
            inputs.append(self.tensor(name, [value], datatype))

        self._append(inputs, self._words_tensor("stop_words", params.stop))
        self._append(inputs, self._words_tensor("bad_words", params.bad_words))

        # BOOL flags: only sent when set, so the model's own defaults apply otherwise
        if params.return_log_probs:
            self._append(inputs, self.optional_tensor("return_log_probs", True, DataType.BOOL))  # noqa: FBT003
        if not request.exclude_input_in_output:
            self._append(inputs, self.optional_tensor("exclude_input_in_output", False, DataType.BOOL))  # noqa: FBT003
        if params.ignore_eos:
            self._append(inputs, self._ignore_eos_tensor(request))

        inputs.extend(self.extra_inputs(request))
        return inputs

    def requested_outputs(self, request: GenerateRequest) -> list[str]:
        """Request text_output, plus log-prob outputs the model declares when they were asked for.

        :param request: The request being served; ``params.return_log_probs`` decides whether
            log-prob outputs are requested.
        :return: Names of the output tensors to request.
        """
        outputs = [TEXT_OUTPUT]
        if request.params.return_log_probs:
            outputs.extend(name for name in _LOG_PROB_OUTPUTS if self.supports_output(name))
        return outputs

    # --- response --------------------------------------------------------------
    def parse_result(self, result: InferResult, *, index: int, previous_text: str) -> GenerateChunk:
        """Parse a TensorRT-LLM response, including optional log-prob tensors.

        TRT-LLM streams deltas already, so ``previous_text`` is not used; the final decoupled
        response usually carries an empty ``text_output``.

        :param result: The raw inference result to parse.
        :param index: Zero-based position of this chunk within the stream.
        :param previous_text: Unused; this backend already streams deltas.
        :return: The parsed :class:`~triton_sdk.core.models.GenerateChunk`.
        """
        text = str(result.output(TEXT_OUTPUT).first() or "") if TEXT_OUTPUT in result.outputs else ""
        is_final = bool(result.parameters.get(FINAL_RESPONSE_PARAM, False))
        log_probs = result.outputs.get("output_log_probs")
        cum = result.outputs.get("cum_log_probs")
        return GenerateChunk(
            text=text,
            index=index,
            is_final=is_final,
            log_probs=[float(value) for value in log_probs.data] if log_probs else None,
            cum_log_probs=float(cum.first()) if cum and cum.data else None,
            raw=result.model_dump(),
        )

    # --- helpers ---------------------------------------------------------------
    def _ignore_eos_tensor(self, request: GenerateRequest) -> InferInput | None:
        """Build the end_id tensor that emulates ``ignore_eos`` for TensorRT-LLM.

        :param request: The request being built; if it already sets ``end_id`` via
            ``extra_inputs``, that explicit value takes precedence and no tensor is built here.
        :return: An ``end_id`` tensor set to :data:`IGNORE_EOS_END_ID`, or ``None`` when the
            model does not declare an ``end_id`` input or the caller set one explicitly.
        """
        if "end_id" in request.extra_inputs:
            logger.debug("ignore_eos requested but extra_inputs sets end_id explicitly; keeping the explicit value")
            return None
        tensor = self.optional_tensor("end_id", IGNORE_EOS_END_ID, DataType.INT32)
        if tensor is None:
            logger.warning("model %s has no end_id input; ignore_eos cannot be honoured", self._model_name)
        return tensor

    def _words_tensor(self, name: str, words: list[str]) -> InferInput | None:
        """Build a BYTES words-list tensor (stop_words / bad_words), if non-empty and declared.

        :param name: Tensor name (e.g. ``"stop_words"``).
        :param words: The words to include.
        :return: The constructed tensor, or ``None`` when ``words`` is empty or the model does
            not declare an input with this name.
        """
        if not words:
            return None
        if not self.supports_input(name):
            logger.debug("model %s has no input %r; dropping %d words", self._model_name, name, len(words))
            return None
        return self.tensor(name, list(words), DataType.BYTES)

    @staticmethod
    def _append(inputs: list[InferInput], tensor: InferInput | None) -> None:
        """Append a tensor to a list in place, skipping ``None``.

        :param inputs: List to append to, mutated in place.
        :param tensor: Tensor to append, or ``None`` to skip.
        """
        if tensor is not None:
            inputs.append(tensor)
