import pytest


pytest.importorskip("grpc", reason="grpc extra not installed")

from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import DataType, InferInput
from triton_sdk.transport import grpc_codec
from triton_sdk.transport.protos import grpc_service_pb2 as pb
from triton_sdk.transport.protos import model_config_pb2 as mc


@pytest.mark.parametrize(
    ("datatype", "values"),
    [
        (DataType.BOOL, [True, False, True]),
        (DataType.INT32, [1, -2, 2**31 - 1]),
        (DataType.INT64, [2**40, -1]),
        (DataType.UINT32, [0, 2**32 - 1]),
        (DataType.FP32, [0.5, -1.25]),
        (DataType.FP64, [3.141592653589793]),
        (DataType.FP16, [1.5, -0.25]),
        (DataType.BF16, [1.0, -2.0, 0.5]),
        (DataType.BYTES, ["hello", "мир", ""]),
    ],
)
def test_tensor_roundtrip(datatype: DataType, values: list[object]) -> None:
    assert grpc_codec.decode_tensor(datatype, grpc_codec.encode_tensor(datatype, values)) == values


def test_build_infer_request_uses_raw_contents_and_parameters() -> None:
    inputs = [InferInput.strings("text_input", ["hi"]), InferInput.scalar("max_tokens", 8, DataType.INT32)]
    request = grpc_codec.build_infer_request(
        "m", inputs, outputs=["text_output"], version="3", request_id="r1", parameters={"priority": 2, "trace": True}
    )
    assert (request.model_name, request.model_version, request.id) == ("m", "3", "r1")
    assert [tensor.name for tensor in request.inputs] == ["text_input", "max_tokens"]
    assert request.inputs[1].datatype == "INT32"
    assert request.inputs[1].shape == [1, 1]
    assert request.raw_input_contents[1] == (8).to_bytes(4, "little")
    assert [tensor.name for tensor in request.outputs] == ["text_output"]
    assert request.parameters["priority"].int64_param == 2  # noqa: PLR2004
    assert request.parameters["trace"].bool_param is True


def test_parse_infer_response_from_raw_contents() -> None:
    response = pb.ModelInferResponse(model_name="m", model_version="1", id="r1")
    response.outputs.add(name="text_output", datatype="BYTES", shape=[1])
    response.outputs.add(name="cum_log_probs", datatype="FP32", shape=[1])
    response.raw_output_contents.extend(
        [grpc_codec.encode_tensor(DataType.BYTES, ["Hello"]), grpc_codec.encode_tensor(DataType.FP32, [-0.5])]
    )
    response.parameters[FINAL_RESPONSE_PARAM].bool_param = True

    result = grpc_codec.parse_infer_response(response)
    assert result.output("text_output").first() == "Hello"
    assert result.output("cum_log_probs").first() == pytest.approx(-0.5)
    assert result.parameters == {FINAL_RESPONSE_PARAM: True}
    assert (result.model_version, result.request_id) == ("1", "r1")


def test_parse_infer_response_from_typed_contents() -> None:
    response = pb.ModelInferResponse(model_name="m")
    tensor = response.outputs.add(name="text_output", datatype="BYTES", shape=[1])
    tensor.contents.bytes_contents.append(b"typed")
    assert grpc_codec.parse_infer_response(response).output("text_output").first() == "typed"


def test_parse_model_config_lifts_decoupled() -> None:
    config = mc.ModelConfig(name="vllm_model", backend="vllm")
    config.model_transaction_policy.decoupled = True
    tensor = config.input.add()
    tensor.name = "text_input"
    tensor.data_type = mc.TYPE_STRING
    tensor.dims.extend([1])

    parsed = grpc_codec.parse_model_config(pb.ModelConfigResponse(config=config))
    assert parsed.decoupled is True
    assert parsed.backend == "vllm"
    assert parsed.input_names == {"text_input"}
