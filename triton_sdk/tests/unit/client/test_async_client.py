from triton_sdk import AsyncTritonLLMClient, SamplingParams
from triton_sdk.core.models import DataType, InferOutput, InferResult, ModelConfig
from triton_sdk.settings import TritonSettings


def _result(text: str, *, final: bool = False) -> InferResult:
    return InferResult(
        model_name="vllm_model",
        outputs={"text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=[text])},
        parameters={"triton_final_response": final},
    )


async def test_generate_uses_detected_backend(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("Hello!", final=True)])
    async with AsyncTritonLLMClient(settings=settings, transport=transport) as client:
        response = await client.generate("hi", SamplingParams(max_tokens=4), model="vllm_model")
    assert response.text == "Hello!"
    assert transport.calls[0]["model"] == "vllm_model"
    assert transport.closed is True


async def test_stream_yields_deltas(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("He"), _result("llo"), _result("", final=True)])
    async with AsyncTritonLLMClient(settings=settings, transport=transport) as client:
        chunks = [chunk async for chunk in client.stream("hi", model="vllm_model")]
    assert [chunk.text for chunk in chunks] == ["He", "llo", ""]
    assert chunks[-1].is_final is True


async def test_generate_uses_stream_api_for_decoupled_models(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    # Triton rejects unary infer for decoupled models; generate() must go through stream_infer with stream=False
    assert vllm_config.decoupled is True
    transport = fake_transport_factory(vllm_config, [_result("Hello!", final=True)])
    async with AsyncTritonLLMClient(settings=settings, transport=transport) as client:
        response = await client.generate("hi", model="vllm_model")
    assert response.text == "Hello!"
    assert transport.calls[0]["method"] == "stream_infer"
    sent = {tensor.name: tensor for tensor in transport.calls[0]["inputs"]}
    assert sent["stream"].data == [False]


async def test_generate_uses_unary_infer_for_coupled_models(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    coupled = vllm_config.model_copy(update={"decoupled": False})
    transport = fake_transport_factory(coupled, [_result("Hello!", final=True)])
    async with AsyncTritonLLMClient(settings=settings, transport=transport) as client:
        await client.generate("hi", model="vllm_model")
    assert transport.calls[0]["method"] == "infer"


async def test_generate_forwards_extra_inputs_and_version(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("ok", final=True)])
    async with AsyncTritonLLMClient(settings=settings, transport=transport) as client:
        await client.generate(
            "hi", model="vllm_model", model_version="2", request_id="r1", extra_inputs={"image": ["data:..."]}
        )
    call = transport.calls[0]
    assert (call["version"], call["request_id"]) == ("2", "r1")
    sent = {tensor.name: tensor for tensor in call["inputs"]}
    assert sent["image"].datatype is DataType.BYTES  # declared TYPE_STRING in the vllm config fixture
