"""Minimal KServe v2 tensor representation shared by HTTP (JSON) and gRPC transports."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DataType(StrEnum):
    """Triton tensor datatypes as spelled in KServe v2 JSON."""

    BOOL = "BOOL"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BF16 = "BF16"
    BYTES = "BYTES"

    @classmethod
    def from_config(cls, data_type: str) -> DataType:
        """Map a config.pbtxt data_type to the KServe spelling.

        :param data_type: Raw value from ``config.pbtxt``, e.g. ``"TYPE_INT32"`` or ``"TYPE_STRING"``.
        :return: The matching :class:`DataType` member (``TYPE_STRING`` maps to :attr:`BYTES`).
        :raises ValueError: If the ``TYPE_`` suffix does not match a known datatype.
        """
        name = data_type.removeprefix("TYPE_")
        return cls.BYTES if name == "STRING" else cls(name)

    @classmethod
    def guess(cls, values: list[Any]) -> DataType:
        """Best-effort datatype for JSON values whose tensor type is unknown.

        Used for generate-endpoint responses, where output tensors are flattened into plain JSON
        and their KServe datatype is not sent by the server.

        :param values: Sample values to inspect; the first non-``None`` entry decides the type.
        :return: :attr:`BOOL` / :attr:`INT64` / :attr:`FP32` for the matching Python type, else :attr:`BYTES`.
        """
        sample = next((value for value in values if value is not None), None)
        if isinstance(sample, bool):
            return cls.BOOL
        if isinstance(sample, int):
            return cls.INT64
        if isinstance(sample, float):
            return cls.FP32
        return cls.BYTES


class InferInput(BaseModel):
    """One KServe v2 input tensor: name, datatype, shape and its data payload."""

    name: str
    datatype: DataType
    shape: list[int]
    data: list[Any]

    @classmethod
    def scalar(cls, name: str, value: Any, datatype: DataType) -> InferInput:
        """Build a single-element tensor with shape ``[1, 1]``.

        :param name: Tensor name as declared in the model config.
        :param value: The scalar value to wrap.
        :param datatype: KServe datatype of ``value``.
        :return: The constructed :class:`InferInput`.
        """
        return cls(name=name, datatype=datatype, shape=[1, 1], data=[value])

    @classmethod
    def strings(cls, name: str, values: list[str]) -> InferInput:
        """Build a BYTES tensor of shape ``[1, len(values)]`` from a list of strings.

        :param name: Tensor name as declared in the model config.
        :param values: String values to store in the tensor.
        :return: The constructed :class:`InferInput` with datatype :attr:`DataType.BYTES`.
        """
        return cls(name=name, datatype=DataType.BYTES, shape=[1, len(values)], data=values)

    def to_kserve_json(self) -> dict[str, Any]:
        """Serialize this tensor into the KServe v2 HTTP JSON shape.

        :return: A dict with ``name``, ``datatype``, ``shape`` and ``data`` keys.
        """
        return {"name": self.name, "datatype": self.datatype.value, "shape": self.shape, "data": self.data}


class InferOutput(BaseModel):
    """One KServe v2 output tensor: name, datatype, shape and its data payload."""

    name: str
    datatype: DataType
    shape: list[int]
    data: list[Any]

    def first(self) -> Any:
        """Return the first element of the tensor's data, or ``None`` if it is empty.

        :return: The first value in :attr:`data`, or ``None``.
        """
        return self.data[0] if self.data else None


class InferResult(BaseModel):
    """Unified result of ModelInfer / ModelStreamInfer / POST infer."""

    model_name: str
    model_version: str | None = None
    request_id: str | None = None
    outputs: dict[str, InferOutput]
    parameters: dict[str, Any] = Field(default_factory=dict)

    def output(self, name: str) -> InferOutput:
        """Look up one output tensor by name.

        :param name: Output tensor name as declared in the model config.
        :return: The matching :class:`InferOutput`.
        :raises KeyError: If no output tensor with that name is present.
        """
        return self.outputs[name]
