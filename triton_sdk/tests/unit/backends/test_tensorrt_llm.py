from typing import Any

import pytest

from triton_sdk.backends import TensorRTLLMAdapter, detect_backend
from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import (
    DataType,
    GenerateRequest,
    GenerateResponse,
    InferInput,
    InferOutput,
    InferResult,
    ModelConfig,
    SamplingParams,
)


def _by_name(inputs: list[InferInput]) -> dict[str, InferInput]:
    return {tensor.name: tensor for tensor in inputs}


def _config(*input_names: str, outputs: tuple[str, ...] = (), max_batch_size: int = 64) -> ModelConfig:
    return ModelConfig(
        name="ensemble",
        platform="ensemble",
        max_batch_size=max_batch_size,
        input=[{"name": name} for name in input_names],
        output=[{"name": name} for name in outputs],
    )


FULL_PARAMS = SamplingParams(
    max_tokens=16,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1,
    presence_penalty=0.2,
    frequency_penalty=0.3,
    length_penalty=1.0,
    min_tokens=2,
    beam_width=1,
    seed=42,
    stop=["\n\n", "###"],
    bad_words=["badword"],
    return_log_probs=True,
)


# --- build_inputs: canonical schema (no config) ------------------------------------


def test_full_mapping_without_config_uses_canonical_names_and_batched_shapes() -> None:
    request = GenerateRequest(model="ensemble", prompt="hi", params=FULL_PARAMS, stream=True)
    inputs = _by_name(TensorRTLLMAdapter().build_inputs(request))

    expected: dict[str, tuple[DataType, list[Any]]] = {
        "text_input": (DataType.BYTES, ["hi"]),
        "max_tokens": (DataType.INT32, [16]),
        "stream": (DataType.BOOL, [True]),
        "temperature": (DataType.FP32, [0.7]),
        "top_p": (DataType.FP32, [0.9]),
        "top_k": (DataType.INT32, [40]),
        "repetition_penalty": (DataType.FP32, [1.1]),
        "presence_penalty": (DataType.FP32, [0.2]),
        "frequency_penalty": (DataType.FP32, [0.3]),
        "length_penalty": (DataType.FP32, [1.0]),
        "min_tokens": (DataType.INT32, [2]),
        "beam_width": (DataType.INT32, [1]),
        "random_seed": (DataType.UINT64, [42]),
        "stop_words": (DataType.BYTES, ["\n\n", "###"]),
        "bad_words": (DataType.BYTES, ["badword"]),
        "return_log_probs": (DataType.BOOL, [True]),
    }
    assert set(inputs) == set(expected)
    for name, (datatype, data) in expected.items():
        assert inputs[name].datatype is datatype, name
        assert inputs[name].data == data, name
        assert inputs[name].shape == [1, len(data)], name  # ensemble batches -> leading batch dim


def test_unset_knobs_are_not_sent() -> None:
    request = GenerateRequest(model="ensemble", prompt="hi", params=SamplingParams(max_tokens=8))
    inputs = _by_name(TensorRTLLMAdapter().build_inputs(request))
    assert set(inputs) == {"text_input", "max_tokens", "stream"}
    assert inputs["stream"].data == [False]


def test_exclude_input_in_output_false_is_sent_explicitly() -> None:
    request = GenerateRequest(model="ensemble", prompt="hi", exclude_input_in_output=False)
    inputs = _by_name(TensorRTLLMAdapter().build_inputs(request))
    assert inputs["exclude_input_in_output"].data == [False]


# --- build_inputs: schema-aware (with config) ------------------------------------------


def test_config_filters_unknown_inputs() -> None:
    config = _config("text_input", "max_tokens", "stream", "temperature")
    request = GenerateRequest(model="ensemble", prompt="hi", params=FULL_PARAMS)
    inputs = _by_name(TensorRTLLMAdapter(config).build_inputs(request))
    assert set(inputs) == {"text_input", "max_tokens", "stream", "temperature"}


def test_min_length_alias_for_older_backends() -> None:
    old = _config("text_input", "max_tokens", "min_length")
    new = _config("text_input", "max_tokens", "min_tokens")
    request = GenerateRequest(model="ensemble", prompt="hi", params=SamplingParams(min_tokens=3))
    assert "min_length" in _by_name(TensorRTLLMAdapter(old).build_inputs(request))
    assert "min_tokens" in _by_name(TensorRTLLMAdapter(new).build_inputs(request))


def test_unbatched_model_gets_flat_shapes() -> None:
    config = _config("text_input", "max_tokens", "stop_words", max_batch_size=0)
    request = GenerateRequest(model="bls", prompt="hi", params=SamplingParams(stop=["a", "b"]))
    inputs = _by_name(TensorRTLLMAdapter(config).build_inputs(request))
    assert inputs["text_input"].shape == [1]
    assert inputs["stop_words"].shape == [2]


def test_extra_inputs_pass_through() -> None:
    request = GenerateRequest(
        model="ensemble",
        prompt="hi",
        extra_inputs={
            "end_id": 2,
            "embedding_bias_words": ["foo", "bar"],
            "custom": InferInput(name="custom", datatype=DataType.FP16, shape=[1, 1], data=[0.5]),
        },
    )
    inputs = _by_name(TensorRTLLMAdapter().build_inputs(request))
    assert (inputs["end_id"].datatype, inputs["end_id"].data) == (DataType.INT32, [2])  # known TRT-LLM input
    assert (inputs["embedding_bias_words"].datatype, inputs["embedding_bias_words"].shape) == (DataType.BYTES, [1, 2])
    assert inputs["custom"].datatype is DataType.FP16


def test_extra_inputs_take_datatype_from_model_config() -> None:
    config = ModelConfig(
        name="ensemble",
        max_batch_size=8,
        input=[
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [-1]},
            {"name": "max_tokens", "data_type": "TYPE_INT32", "dims": [-1]},
            {"name": "my_flag", "data_type": "TYPE_BOOL", "dims": [1]},
            {"name": "my_weight", "data_type": "TYPE_FP16", "dims": [1]},
        ],
    )
    request = GenerateRequest(
        model="ensemble", prompt="hi", extra_inputs={"my_flag": 1, "my_weight": 0.5, "unknown": 3}
    )
    inputs = _by_name(TensorRTLLMAdapter(config).build_inputs(request))
    assert inputs["my_flag"].datatype is DataType.BOOL  # config wins over the JSON type (int)
    assert inputs["my_weight"].datatype is DataType.FP16
    assert inputs["unknown"].datatype is DataType.INT64  # not declared anywhere -> guessed


def test_registry_passes_config_to_adapter(trtllm_config: ModelConfig) -> None:
    adapter = detect_backend(trtllm_config)
    assert isinstance(adapter, TensorRTLLMAdapter)
    assert adapter.config is trtllm_config
    assert adapter.batched is True


# --- requested_outputs -----------------------------------------------------------------


def test_requested_outputs_include_logprobs_when_model_declares_them() -> None:
    with_probs = _config("text_input", "max_tokens", outputs=("text_output", "cum_log_probs", "output_log_probs"))
    without = _config("text_input", "max_tokens", outputs=("text_output",))
    request = GenerateRequest(model="ensemble", prompt="hi", params=SamplingParams(return_log_probs=True))
    assert TensorRTLLMAdapter(with_probs).requested_outputs(request) == [
        "text_output",
        "cum_log_probs",
        "output_log_probs",
    ]
    assert TensorRTLLMAdapter(without).requested_outputs(request) == ["text_output"]
    assert TensorRTLLMAdapter().requested_outputs(request) == ["text_output", "cum_log_probs", "output_log_probs"]


# --- parse_result ----------------------------------------------------------------------


def _result(text: str, *, final: bool = False, **extra: list[Any]) -> InferResult:
    outputs = {"text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=[text])}
    for name, data in extra.items():
        outputs[name] = InferOutput(name=name, datatype=DataType.FP32, shape=[len(data)], data=data)
    return InferResult(model_name="ensemble", outputs=outputs, parameters={FINAL_RESPONSE_PARAM: final})


def test_parse_result_keeps_deltas_and_extracts_logprobs() -> None:
    adapter = TensorRTLLMAdapter()
    chunk = adapter.parse_result(
        _result(" world", final=True, cum_log_probs=[-1.25], output_log_probs=[-0.5, -0.75]),
        index=1,
        previous_text="Hello",
    )
    assert chunk.text == " world"  # TRT-LLM already streams deltas: previous_text is ignored
    assert chunk.is_final is True
    assert chunk.cum_log_probs == pytest.approx(-1.25)
    assert chunk.log_probs == pytest.approx([-0.5, -0.75])


def test_parse_result_tolerates_final_response_without_text_output() -> None:
    result = InferResult(model_name="ensemble", outputs={}, parameters={FINAL_RESPONSE_PARAM: True})
    chunk = TensorRTLLMAdapter().parse_result(result, index=3, previous_text="abc")
    assert (chunk.text, chunk.is_final) == ("", True)


def test_response_from_chunks_carries_cum_log_probs() -> None:
    adapter = TensorRTLLMAdapter()
    chunks = [
        adapter.parse_result(_result("Hel", cum_log_probs=[-0.1]), index=0, previous_text=""),
        adapter.parse_result(_result("lo", final=True, cum_log_probs=[-0.4]), index=1, previous_text="Hel"),
    ]
    response = GenerateResponse.from_chunks("ensemble", chunks)
    assert (response.text, response.cum_log_probs) == ("Hello", pytest.approx(-0.4))


# --- ignore_eos ------------------------------------------------------------------------


def test_ignore_eos_sends_end_id_minus_one() -> None:
    request = GenerateRequest(model="ensemble", prompt="hi", params=SamplingParams(ignore_eos=True))
    inputs = _by_name(TensorRTLLMAdapter().build_inputs(request))
    assert (inputs["end_id"].datatype, inputs["end_id"].data) == (DataType.INT32, [-1])


def test_ignore_eos_yields_to_explicit_end_id() -> None:
    request = GenerateRequest(
        model="ensemble", prompt="hi", params=SamplingParams(ignore_eos=True), extra_inputs={"end_id": 151643}
    )
    inputs = TensorRTLLMAdapter().build_inputs(request)
    end_ids = [tensor for tensor in inputs if tensor.name == "end_id"]
    assert len(end_ids) == 1
    assert end_ids[0].data == [151643]


def test_ignore_eos_dropped_when_model_has_no_end_id(caplog: pytest.LogCaptureFixture) -> None:
    config = _config("text_input", "max_tokens")
    request = GenerateRequest(model="ensemble", prompt="hi", params=SamplingParams(ignore_eos=True))
    with caplog.at_level("WARNING"):
        inputs = _by_name(TensorRTLLMAdapter(config).build_inputs(request))
    assert "end_id" not in inputs
    assert "ignore_eos cannot be honoured" in caplog.text
