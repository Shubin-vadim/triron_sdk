from triton_sdk.chat import ChatMessage
from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import DataType, InferOutput, InferResult, ModelConfig
from triton_sdk.openai import AsyncOpenAICompatClient, ChatCompletion, ChatCompletionChunk, OpenAICompatClient
from triton_sdk.settings import TritonSettings


class WordTemplate:
    """Deterministic template so usage is checkable: one token per whitespace-separated word."""

    def render(self, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
        return " ".join(f"{message.role.value}: {message.content}" for message in messages)

    def count_tokens(self, text: str) -> int | None:
        return len(text.split())


def _result(text: str, *, final: bool = False) -> InferResult:
    return InferResult(
        model_name="vllm_model",
        outputs={"text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=[text])},
        parameters={FINAL_RESPONSE_PARAM: final},
    )


MESSAGES = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi there"}]


async def test_chat_completion_shape(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("Hello back", final=True)])
    async with AsyncOpenAICompatClient(settings=settings, transport=transport, chat_template=WordTemplate()) as client:
        response = await client.chat.completions.create(model="vllm_model", messages=MESSAGES, max_tokens=16)

    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert response.id.startswith("chatcmpl-")
    assert response.model == "vllm_model"
    choice = response.choices[0]
    assert (choice.index, choice.message.role, choice.message.content) == (0, "assistant", "Hello back")
    assert choice.finish_reason == "stop"
    assert response.usage is not None
    # prompt = "system: be brief user: hi there" -> 6 words, completion 2 words
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (6, 2, 8)
    # the prompt went through the chat template, vLLM adapter carried the openai params
    sent = {tensor.name: tensor for tensor in transport.calls[0]["inputs"]}
    assert sent["text_input"].data == ["system: be brief user: hi there"]
    assert '"max_tokens": 16' in sent["sampling_parameters"].data[0]


async def test_chat_completion_stream_mirrors_openai_chunk_sequence(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("Hel"), _result("lo"), _result("", final=True)])
    async with AsyncOpenAICompatClient(settings=settings, transport=transport, chat_template=WordTemplate()) as client:
        stream = await client.chat.completions.create(
            model="vllm_model", messages=MESSAGES, stream=True, stream_options={"include_usage": True}
        )
        chunks = [chunk async for chunk in stream]

    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 1  # same id across the stream, like openai
    assert chunks[0].choices[0].delta.role == "assistant"
    assert [chunk.choices[0].delta.content for chunk in chunks[1:3]] == ["Hel", "lo"]
    final = chunks[3]
    assert (final.choices[0].delta.content, final.choices[0].finish_reason) == (None, "stop")
    usage_chunk = chunks[4]
    assert usage_chunk.choices == []
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.completion_tokens == 1


async def test_usage_is_none_without_tokenizer(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("ok", final=True)])
    async with AsyncOpenAICompatClient(settings=settings, transport=transport) as client:  # SimpleChatTemplate
        response = await client.chat.completions.create(model="vllm_model", messages=MESSAGES)
    assert response.usage is None
    assert response.choices[0].finish_reason == "stop"


async def test_models_list(settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type) -> None:
    async with AsyncOpenAICompatClient(settings=settings, transport=fake_transport_factory(vllm_config)) as client:
        models = await client.models.list()
    assert models.object == "list"
    assert [model.id for model in models.data] == ["vllm_model"]
    assert models.data[0].owned_by == "triton"


def test_sync_client_chat_and_stream(
    settings: TritonSettings, vllm_config: ModelConfig, fake_transport_factory: type
) -> None:
    transport = fake_transport_factory(vllm_config, [_result("a"), _result(" b", final=True)])
    with OpenAICompatClient(settings=settings, transport=transport, chat_template=WordTemplate()) as client:
        response = client.chat.completions.create(model="vllm_model", messages=MESSAGES)
        assert response.choices[0].message.content == "a b"  # decoupled model: collected over the stream API

        chunks = list(client.chat.completions.create(model="vllm_model", messages=MESSAGES, stream=True))
        texts = [chunk.choices[0].delta.content for chunk in chunks]
        assert texts == ["", "a", " b", None]
        assert chunks[-1].choices[0].finish_reason == "stop"

        completion = client.completions.create(model="vllm_model", prompt="raw prompt", max_tokens=4)
        assert completion.object == "text_completion"
        assert completion.choices[0].text == "a b"
