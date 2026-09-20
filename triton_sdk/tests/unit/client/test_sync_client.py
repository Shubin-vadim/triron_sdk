from triton_sdk import TritonLLMClient
from triton_sdk.core.models import DataType, InferOutput, InferResult, ModelConfig
from triton_sdk.settings import TritonSettings


def test_sync_generate(settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type) -> None:
    result = InferResult(
        model_name="vllm_model",
        outputs={"text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=["ok"])},
        parameters={"triton_final_response": True},
    )
    with TritonLLMClient(settings=settings, transport=fake_transport_factory(vllm_config, [result])) as client:
        assert client.generate("hi", model="vllm_model").text == "ok"
