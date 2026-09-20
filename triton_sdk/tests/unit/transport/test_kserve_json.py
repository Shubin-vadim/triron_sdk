import pytest

from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import DataType, InferInput
from triton_sdk.exceptions import InferenceError
from triton_sdk.transport.kserve_json import (
    build_generate_body,
    build_infer_body,
    mark_final,
    parse_generate_event,
    parse_infer_response,
)


def _inputs() -> list[InferInput]:
    return [
        InferInput.strings("text_input", ["hello"]),
        InferInput.scalar("stream", True, DataType.BOOL),  # noqa: FBT003
        InferInput.strings("stop_words", ["\n", "###"]),
    ]


def test_build_infer_body_is_kserve_v2() -> None:
    body = build_infer_body(_inputs(), outputs=["text_output"], request_id="r1", parameters={"priority": 1})
    assert body["id"] == "r1"
    assert body["parameters"] == {"priority": 1}
    assert body["outputs"] == [{"name": "text_output"}]
    assert body["inputs"][0] == {"name": "text_input", "datatype": "BYTES", "shape": [1, 1], "data": ["hello"]}


def test_build_generate_body_flattens_tensors() -> None:
    body = build_generate_body(_inputs(), request_id="r1")
    assert body == {"text_input": "hello", "stream": True, "stop_words": ["\n", "###"], "id": "r1"}


def test_parse_infer_response() -> None:
    payload = {
        "model_name": "m",
        "model_version": "1",
        "id": "r1",
        "parameters": {FINAL_RESPONSE_PARAM: True},
        "outputs": [{"name": "text_output", "datatype": "BYTES", "shape": [1], "data": ["hi"]}],
    }
    result = parse_infer_response(payload, model="m")
    assert result.output("text_output").first() == "hi"
    assert result.output("text_output").datatype is DataType.BYTES
    assert result.request_id == "r1"
    assert result.parameters[FINAL_RESPONSE_PARAM] is True


def test_parse_infer_response_raises_on_error_payload() -> None:
    with pytest.raises(InferenceError, match="boom"):
        parse_infer_response({"error": "boom"}, model="m")


def test_parse_generate_event_guesses_datatypes_and_lifts_final_flag() -> None:
    event = {
        "model_name": "m",
        "model_version": "1",
        "text_output": "Hello",
        "cum_log_probs": -1.5,
        "sequence_length": 3,
        FINAL_RESPONSE_PARAM: True,
    }
    result = parse_generate_event(event, model="m")
    assert set(result.outputs) == {"text_output", "cum_log_probs", "sequence_length"}
    assert result.output("cum_log_probs").datatype is DataType.FP32
    assert result.output("sequence_length").datatype is DataType.INT64
    assert result.parameters == {FINAL_RESPONSE_PARAM: True}


def test_parse_generate_event_raises_on_error() -> None:
    with pytest.raises(InferenceError, match="CUDA"):
        parse_generate_event({"error": "CUDA out of memory"}, model="m")


def test_mark_final_is_idempotent() -> None:
    result = parse_generate_event({"text_output": "x"}, model="m")
    marked = mark_final(result)
    assert marked.parameters[FINAL_RESPONSE_PARAM] is True
    assert mark_final(marked) is marked
    assert FINAL_RESPONSE_PARAM not in result.parameters  # original untouched
