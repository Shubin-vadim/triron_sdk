"""SDK tensor models <-> Triton gRPC protobuf messages. No network code; numpy-free.

Inputs are sent as raw_input_contents (little-endian packed), which is what Triton itself prefers and the
only encoding that covers FP16/BF16. Outputs are accepted in either raw_output_contents or typed contents.
"""

from __future__ import annotations

import struct
from typing import Any

from google.protobuf.json_format import MessageToDict

from ..core.models import (
    DataType,
    InferInput,
    InferOutput,
    InferResult,
    ModelConfig,
    ModelMetadata,
    RepositoryModel,
    ServerMetadata,
    TensorMetadata,
)
from ..core.utils import decode_bytes_tensor, encode_bytes_tensor
from .protos import grpc_service_pb2 as pb


# struct format codes for fixed-width datatypes (little-endian applied at pack time)
_STRUCT_CODES: dict[DataType, str] = {
    DataType.BOOL: "?",
    DataType.UINT8: "B",
    DataType.UINT16: "H",
    DataType.UINT32: "I",
    DataType.UINT64: "Q",
    DataType.INT8: "b",
    DataType.INT16: "h",
    DataType.INT32: "i",
    DataType.INT64: "q",
    DataType.FP16: "e",
    DataType.FP32: "f",
    DataType.FP64: "d",
}

# typed-contents field per datatype, for servers/backends that fill InferTensorContents instead of raw bytes
_CONTENTS_FIELDS: dict[DataType, str] = {
    DataType.BOOL: "bool_contents",
    DataType.INT8: "int_contents",
    DataType.INT16: "int_contents",
    DataType.INT32: "int_contents",
    DataType.INT64: "int64_contents",
    DataType.UINT8: "uint_contents",
    DataType.UINT16: "uint_contents",
    DataType.UINT32: "uint_contents",
    DataType.UINT64: "uint64_contents",
    DataType.FP32: "fp32_contents",
    DataType.FP64: "fp64_contents",
    DataType.BYTES: "bytes_contents",
}


# --- tensors -----------------------------------------------------------------


def encode_tensor(datatype: DataType, values: list[Any]) -> bytes:
    """Pack tensor values into the little-endian bytes Triton expects in raw_input_contents.

    :param datatype: KServe datatype of the values.
    :param values: Values to encode, in tensor order.
    :return: The packed binary payload.
    :raises KeyError: If ``datatype`` is not one of the supported fixed-width types (see
        :data:`_STRUCT_CODES`) and is not :attr:`~triton_sdk.core.models.DataType.BYTES` or
        :attr:`~triton_sdk.core.models.DataType.BF16`.
    """
    if datatype is DataType.BYTES:
        return encode_bytes_tensor([value if isinstance(value, str) else str(value) for value in values])
    if datatype is DataType.BF16:
        # bf16 is the upper half of fp32
        return b"".join(struct.pack("<f", float(value))[2:] for value in values)
    return struct.pack(f"<{len(values)}{_STRUCT_CODES[datatype]}", *values)


def decode_tensor(datatype: DataType, raw: bytes) -> list[Any]:
    """Unpack a tensor's raw_output_contents payload into Python values.

    :param datatype: KServe datatype of the payload.
    :param raw: Binary payload as received from the server.
    :return: The decoded values, in tensor order.
    :raises KeyError: If ``datatype`` is not one of the supported fixed-width types (see
        :data:`_STRUCT_CODES`) and is not :attr:`~triton_sdk.core.models.DataType.BYTES` or
        :attr:`~triton_sdk.core.models.DataType.BF16`.
    """
    if datatype is DataType.BYTES:
        return decode_bytes_tensor(raw)
    if datatype is DataType.BF16:
        return [struct.unpack("<f", b"\x00\x00" + raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
    code = _STRUCT_CODES[datatype]
    count = len(raw) // struct.calcsize(code)
    return list(struct.unpack(f"<{count}{code}", raw))


def to_proto_inputs(inputs: list[InferInput]) -> tuple[list[pb.ModelInferRequest.InferInputTensor], list[bytes]]:
    """Convert SDK input tensors into protobuf tensor metadata and their raw content bytes.

    :param inputs: Input tensors to convert.
    :return: A tuple of (protobuf tensor descriptors, encoded ``raw_input_contents`` payloads),
        aligned by index.
    """
    tensors = [
        pb.ModelInferRequest.InferInputTensor(name=tensor.name, datatype=tensor.datatype.value, shape=tensor.shape)
        for tensor in inputs
    ]
    raw = [encode_tensor(tensor.datatype, tensor.data) for tensor in inputs]
    return tensors, raw


def to_proto_parameters(parameters: dict[str, Any] | None) -> dict[str, pb.InferParameter]:
    """Convert a plain parameters mapping into protobuf ``InferParameter`` messages.

    :param parameters: Mapping of parameter name to a ``bool``, ``int``, ``float`` or any other
        value (stringified as a fallback).
    :return: A mapping of the same keys to their ``InferParameter`` protobuf representation.
    """
    result: dict[str, pb.InferParameter] = {}
    for key, value in (parameters or {}).items():
        if isinstance(value, bool):
            result[key] = pb.InferParameter(bool_param=value)
        elif isinstance(value, int):
            result[key] = pb.InferParameter(int64_param=value)
        elif isinstance(value, float):
            result[key] = pb.InferParameter(double_param=value)
        else:
            result[key] = pb.InferParameter(string_param=str(value))
    return result


def from_proto_parameters(parameters: Any) -> dict[str, Any]:
    """Convert a protobuf parameters map back into plain Python values.

    :param parameters: Protobuf map of ``InferParameter`` messages (e.g. ``response.parameters``).
    :return: A mapping of the same keys to their unwrapped Python value.
    """
    result: dict[str, Any] = {}
    for key, param in parameters.items():
        field = param.WhichOneof("parameter_choice")
        result[key] = getattr(param, field) if field else None
    return result


def build_infer_request(
    model: str,
    inputs: list[InferInput],
    *,
    outputs: list[str] | None = None,
    version: str | None = None,
    request_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> pb.ModelInferRequest:
    """Build a ``ModelInferRequest`` protobuf message for ModelInfer / ModelStreamInfer.

    :param model: Model name.
    :param inputs: Input tensors to include.
    :param outputs: Names of output tensors to request; none are requested when omitted.
    :param version: Model version; the server default version when omitted.
    :param request_id: Caller-supplied id to include as the request's ``id`` field.
    :param parameters: Extra request-level parameters to attach.
    :return: The constructed ``ModelInferRequest``.
    """
    tensors, raw = to_proto_inputs(inputs)
    request = pb.ModelInferRequest(
        model_name=model,
        model_version=version or "",
        id=request_id or "",
        inputs=tensors,
        raw_input_contents=raw,
        outputs=[pb.ModelInferRequest.InferRequestedOutputTensor(name=name) for name in outputs or []],
    )
    for key, param in to_proto_parameters(parameters).items():
        request.parameters[key].CopyFrom(param)
    return request


def parse_infer_response(response: pb.ModelInferResponse) -> InferResult:
    """Parse a ``ModelInferResponse`` protobuf message into an :class:`InferResult`.

    Output tensors are decoded from ``raw_output_contents`` when present, falling back to typed
    ``contents`` fields otherwise.

    :param response: The response message to parse (from ModelInfer or one ModelStreamInfer frame).
    :return: The parsed :class:`~triton_sdk.core.models.InferResult`.
    """
    outputs: dict[str, InferOutput] = {}
    raw_contents = list(response.raw_output_contents)
    for index, tensor in enumerate(response.outputs):
        datatype = DataType(tensor.datatype)
        if index < len(raw_contents):
            data = decode_tensor(datatype, raw_contents[index])
        else:
            data = list(getattr(tensor.contents, _CONTENTS_FIELDS[datatype]))
            if datatype is DataType.BYTES:
                data = [item.decode("utf-8") if isinstance(item, bytes) else item for item in data]
        outputs[tensor.name] = InferOutput(name=tensor.name, datatype=datatype, shape=list(tensor.shape), data=data)
    return InferResult(
        model_name=response.model_name,
        model_version=response.model_version or None,
        request_id=response.id or None,
        outputs=outputs,
        parameters=from_proto_parameters(response.parameters),
    )


# --- metadata ----------------------------------------------------------------


def parse_model_metadata(response: pb.ModelMetadataResponse) -> ModelMetadata:
    """Parse a ``ModelMetadataResponse`` protobuf message into a :class:`ModelMetadata`.

    :param response: The response message to parse.
    :return: The parsed :class:`~triton_sdk.core.models.ModelMetadata`.
    """

    def tensors(items: Any) -> list[TensorMetadata]:
        """Convert a repeated protobuf tensor-metadata field into :class:`TensorMetadata` instances.

        :param items: Repeated field of protobuf tensor metadata (inputs or outputs).
        :return: The equivalent list of :class:`~triton_sdk.core.models.TensorMetadata`.
        """
        return [TensorMetadata(name=item.name, datatype=item.datatype, shape=list(item.shape)) for item in items]

    return ModelMetadata(
        name=response.name,
        versions=list(response.versions),
        platform=response.platform,
        inputs=tensors(response.inputs),
        outputs=tensors(response.outputs),
    )


def parse_model_config(response: pb.ModelConfigResponse) -> ModelConfig:
    """Parse a ``ModelConfigResponse`` protobuf message into a :class:`ModelConfig`.

    :param response: The response message to parse.
    :return: The parsed :class:`~triton_sdk.core.models.ModelConfig`.
    """
    return ModelConfig.model_validate(MessageToDict(response.config, preserving_proto_field_name=True))


def parse_server_metadata(response: pb.ServerMetadataResponse) -> ServerMetadata:
    """Parse a ``ServerMetadataResponse`` protobuf message into a :class:`ServerMetadata`.

    :param response: The response message to parse.
    :return: The parsed :class:`~triton_sdk.core.models.ServerMetadata`.
    """
    return ServerMetadata(name=response.name, version=response.version, extensions=list(response.extensions))


def parse_repository_index(response: pb.RepositoryIndexResponse) -> list[RepositoryModel]:
    """Parse a ``RepositoryIndexResponse`` protobuf message into a list of :class:`RepositoryModel`.

    :param response: The response message to parse.
    :return: One :class:`~triton_sdk.core.models.RepositoryModel` per entry in ``response.models``.
    """
    return [
        RepositoryModel(
            name=item.name, version=item.version or None, state=item.state or None, reason=item.reason or None
        )
        for item in response.models
    ]
