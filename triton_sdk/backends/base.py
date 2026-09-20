"""Backend adapter contract: GenerateRequest <-> Triton tensors for one model family.

Adapters are schema-aware: given the ModelConfig they emit only the inputs the deployed model declares, pick the
right tensor shapes (batched models want [1, N], max_batch_size == 0 models want [N]) and resolve renamed
tensors between backend versions. Without a config they fall back to the canonical schema of that backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..core.constants import BackendKind
from ..core.models import DataType, GenerateChunk, GenerateRequest, InferInput, InferResult, ModelConfig


logger = logging.getLogger(__name__)


class BackendAdapter(ABC):
    """Translates GenerateRequest/GenerateChunk to and from one Triton backend's tensor schema."""

    kind: ClassVar[BackendKind]
    #: shape convention when no config is available (ensembles/BLS batch, python backends usually do not)
    default_batched: ClassVar[bool] = True
    #: datatypes of well-known optional inputs, used for plain values in request.extra_inputs when no config is present
    known_input_types: ClassVar[dict[str, DataType]] = {}

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Bind the adapter to a model config, if one is available.

        :param config: The deployed model's :class:`~triton_sdk.core.models.ModelConfig`;
            ``None`` falls back to the backend's canonical schema.
        """
        self.config = config

    # --- contract ------------------------------------------------------------
    @classmethod
    @abstractmethod
    def matches(cls, config: ModelConfig) -> bool:
        """Decide whether this adapter understands the model described by ``config``.

        :param config: Model config to inspect.
        :return: ``True`` if this adapter's schema matches the model's declared inputs.
        """

    @abstractmethod
    def build_inputs(self, request: GenerateRequest) -> list[InferInput]:
        """Translate a :class:`GenerateRequest` into KServe v2 input tensors.

        :param request: The request to translate.
        :return: The input tensors to send to Triton.
        """

    @abstractmethod
    def requested_outputs(self, request: GenerateRequest) -> list[str]:
        """List the output tensor names to ask for (e.g. ``text_output``, ``cum_log_probs``).

        :param request: The request being served; may affect which optional outputs are requested.
        :return: Names of the output tensors to request.
        """

    @abstractmethod
    def parse_result(self, result: InferResult, *, index: int, previous_text: str) -> GenerateChunk:
        """Turn one server response into a :class:`GenerateChunk`.

        :param result: The raw inference result to parse.
        :param index: Zero-based position of this chunk within the stream.
        :param previous_text: Text already emitted by earlier chunks; lets adapters whose backend
            streams cumulative text emit deltas instead.
        :return: The parsed :class:`~triton_sdk.core.models.GenerateChunk`.
        """

    def supports_streaming(self) -> bool:
        """Check whether the bound model can be streamed.

        :return: ``True`` if no config is bound, or if the bound config is decoupled.
        """
        return self.config is None or self.config.decoupled

    # --- schema helpers shared by adapters -----------------------------------
    @property
    def batched(self) -> bool:
        """Whether tensors should carry a leading batch dimension.

        :return: ``True`` when the bound config declares ``max_batch_size > 0``, or when no
            config is bound and :attr:`default_batched` is ``True``.
        """
        if self.config is None:
            return self.default_batched
        return self.config.max_batch_size > 0

    def supports_input(self, name: str) -> bool:
        """Check whether the bound model declares a given input tensor.

        :param name: Input tensor name to check.
        :return: ``True`` if no config is bound, or if the config declares that input.
        """
        return self.config is None or name in self.config.input_names

    def supports_output(self, name: str) -> bool:
        """Check whether the bound model declares a given output tensor.

        :param name: Output tensor name to check.
        :return: ``True`` if no config is bound, if the config declares no outputs at all, or if
            the config declares that output.
        """
        return self.config is None or not self.config.output or name in self.config.output_names

    def resolve_input(self, *candidates: str) -> str | None:
        """Pick the first candidate name the model declares, handling renames across versions.

        :param candidates: Candidate input names, in preference order.
        :return: The first candidate found in the bound config's declared inputs; the first
            candidate unconditionally when no config is bound; ``None`` if none match.
        """
        if self.config is None:
            return candidates[0]
        return next((name for name in candidates if name in self.config.input_names), None)

    def tensor(self, name: str, values: list[Any], datatype: DataType) -> InferInput:
        """Build an input tensor with the shape appropriate for the bound model.

        :param name: Tensor name.
        :param values: Tensor data.
        :param datatype: KServe datatype of ``values``.
        :return: The constructed :class:`~triton_sdk.core.models.InferInput`, shaped
            ``[1, len(values)]`` when :attr:`batched`, else ``[len(values)]``.
        """
        shape = [1, len(values)] if self.batched else [len(values)]
        return InferInput(name=name, datatype=datatype, shape=shape, data=values)

    def optional_tensor(self, name: str, value: Any, datatype: DataType) -> InferInput | None:
        """Build a single-value tensor, or skip it if unset or undeclared by the model.

        :param name: Tensor name.
        :param value: Scalar value; ``None`` means "not set".
        :param datatype: KServe datatype of ``value``.
        :return: The constructed tensor, or ``None`` when ``value`` is ``None`` or the bound
            model does not declare an input with this name.
        """
        if value is None:
            return None
        if not self.supports_input(name):
            logger.debug("model %s has no input %r; dropping", self._model_name, name)
            return None
        return self.tensor(name, [value], datatype)

    def extra_inputs(self, request: GenerateRequest) -> list[InferInput]:
        """Build tensors for the escape-hatch ``request.extra_inputs`` mapping.

        Values that are already :class:`InferInput` pass through unchanged. Plain values get
        their datatype from the model config (``data_type``), then from :attr:`known_input_types`,
        and only then guessed from the JSON type of the value.

        :param request: The request whose ``extra_inputs`` to translate.
        :return: The additional input tensors to send.
        """
        inputs: list[InferInput] = []
        for name, value in request.extra_inputs.items():
            if isinstance(value, InferInput):
                inputs.append(value)
                continue
            values = list(value) if isinstance(value, list | tuple) else [value]
            inputs.append(self.tensor(name, values, self.input_datatype(name, values)))
        return inputs

    def input_datatype(self, name: str, values: list[Any]) -> DataType:
        """Determine the datatype to use for a plain (non-``InferInput``) extra input value.

        :param name: Input tensor name.
        :param values: Candidate values, used as a fallback for type guessing.
        :return: The config-declared datatype when available, else the datatype from
            :attr:`known_input_types`, else a best-effort guess from ``values``.
        """
        declared = self.config.input_datatype(name) if self.config else None
        if declared:
            return DataType.from_config(declared)
        return self.known_input_types.get(name) or DataType.guess(values)

    @property
    def _model_name(self) -> str:
        """Model name for log messages.

        :return: The bound config's model name, or ``"<unknown>"`` when no config is bound.
        """
        return self.config.name if self.config else "<unknown>"
