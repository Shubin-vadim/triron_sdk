import json

from triton_sdk.backends import VLLMAdapter
from triton_sdk.core.models import DataType, GenerateRequest, InferOutput, InferResult, SamplingParams


def _result(text: str, *, final: bool = False) -> InferResult:
    return InferResult(
        model_name="vllm_model",
        outputs={"text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=[text])},
        parameters={"triton_final_response": final},
    )


def test_build_inputs_serialises_sampling_parameters() -> None:
    request = GenerateRequest(model="m", prompt="hi", params=SamplingParams(max_tokens=8, temperature=0.1, stop=["\n"]))
    inputs = {tensor.name: tensor for tensor in VLLMAdapter().build_inputs(request)}
    sampling = json.loads(inputs["sampling_parameters"].data[0])
    assert sampling == {"max_tokens": 8, "temperature": 0.1, "stop": ["\n"]}
    assert inputs["text_input"].data == ["hi"]


def test_parse_result_passes_deltas_through() -> None:
    # vllm_backend streams deltas (verified live on Triton 25.08): a delta that happens to start with the text
    # seen so far must NOT be trimmed
    adapter = VLLMAdapter()
    first = adapter.parse_result(_result("a"), index=0, previous_text="")
    second = adapter.parse_result(_result("ab", final=True), index=1, previous_text="a")
    assert (first.text, second.text, second.is_final) == ("a", "ab", True)


def test_parse_result_cumulative_mode_for_legacy_backends() -> None:
    class LegacyVLLM(VLLMAdapter):
        cumulative_text = True

    adapter = LegacyVLLM()
    first = adapter.parse_result(_result("Hello"), index=0, previous_text="")
    second = adapter.parse_result(_result("Hello world", final=True), index=1, previous_text="Hello")
    assert (first.text, second.text) == ("Hello", " world")


def test_vllm_shapes_are_flat_for_unbatched_model() -> None:
    from triton_sdk.core.models import ModelConfig

    config = ModelConfig(name="vllm_model", backend="vllm", max_batch_size=0, input=[{"name": "text_input"}])
    request = GenerateRequest(model="vllm_model", prompt="hi")
    inputs = {tensor.name: tensor for tensor in VLLMAdapter(config).build_inputs(request)}
    assert inputs["text_input"].shape == [1]  # vllm_backend declares max_batch_size: 0
    assert inputs["stream"].shape == [1]
    assert "exclude_input_in_output" not in inputs  # not declared by this config -> dropped


def test_vllm_default_shapes_without_config_are_flat() -> None:
    request = GenerateRequest(model="vllm_model", prompt="hi")
    inputs = {tensor.name: tensor for tensor in VLLMAdapter().build_inputs(request)}
    assert inputs["text_input"].shape == [1]
    assert inputs["exclude_input_in_output"].data == [True]
