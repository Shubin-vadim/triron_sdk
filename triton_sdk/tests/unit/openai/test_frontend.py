"""OpenAICompatClient(protocol="openai") against a mocked Triton OpenAI frontend."""

import json

import httpx
import pytest
import respx

from triton_sdk.core.constants import Protocol
from triton_sdk.exceptions import ModelNotFoundError, TritonError
from triton_sdk.openai import AsyncOpenAICompatClient, ChatCompletion, OpenAICompatClient
from triton_sdk.openai.frontend import AsyncOpenAIFrontend
from triton_sdk.settings import TritonSettings


BASE = "http://localhost:9000"
MESSAGES = [{"role": "user", "content": "hi"}]


def _settings() -> TritonSettings:
    return TritonSettings(  # type: ignore[call-arg]
        URL="localhost:9000",
        PROTOCOL=Protocol.OPENAI,
        TOKEN="secret",  # noqa: S106 - test fixture
        _env_file=None,
    )


def _chat_payload(content: str = "Hello!") -> dict[str, object]:
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "vllm_model",
        "system_fingerprint": None,  # extra fields from the server are tolerated
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


def _sse(*events: object) -> str:
    return "".join(f"data: {event if isinstance(event, str) else json.dumps(event)}\n\n" for event in events)


@respx.mock
async def test_chat_forwards_body_and_parses_server_response() -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, json=_chat_payload()))
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        assert client.native is None
        response = await client.chat.completions.create(
            model="vllm_model", messages=MESSAGES, max_tokens=8, temperature=0.1, extra_body={"top_k": 3}
        )

    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "model": "vllm_model",
        "messages": MESSAGES,
        "max_tokens": 8,
        "temperature": 0.1,
        "top_k": 3,  # extra_body merged into the body, like the openai SDK does
        "stream": False,
    }
    assert route.calls.last.request.headers["authorization"] == "Bearer secret"
    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content == "Hello!"
    assert response.usage is not None
    assert response.usage.total_tokens == 15  # noqa: PLR2004 - server-side usage, not computed by us


@respx.mock
async def test_chat_stream_parses_sse_and_stops_at_done() -> None:
    chunk = {"id": "chatcmpl-abc", "object": "chat.completion.chunk", "created": 1, "model": "vllm_model"}
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=_sse(
                {
                    **chunk,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                },
                {**chunk, "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
                {**chunk, "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
                {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )
    )
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        stream = await client.chat.completions.create(model="vllm_model", messages=MESSAGES, stream=True)
        chunks = [piece async for piece in stream]

    assert [piece.choices[0].delta.content for piece in chunks] == ["", "Hel", "lo", None]
    assert chunks[-1].choices[0].finish_reason == "stop"


@respx.mock
async def test_frontend_errors_map_to_triton_errors() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(404, json={"detail": "Unknown model: nope"})
    )
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        with pytest.raises(ModelNotFoundError, match="Unknown model") as info:
            await client.chat.completions.create(model="nope", messages=MESSAGES)
    assert info.value.model == "nope"


@respx.mock
async def test_frontend_400_carries_server_message() -> None:
    respx.post(f"{BASE}/v1/completions").mock(
        return_value=httpx.Response(400, json={"detail": {"message": "max_tokens must be positive"}})
    )
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        with pytest.raises(TritonError, match="max_tokens must be positive"):
            await client.completions.create(model="vllm_model", prompt="x", max_tokens=-1)


@respx.mock
async def test_models_and_ready() -> None:
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "vllm_model", "object": "model", "created": 1, "owned_by": "Triton Inference Server"}],
            },
        )
    )
    respx.get(f"{BASE}/health/ready").mock(return_value=httpx.Response(200))
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        models = await client.models.list()
        assert [model.id for model in models.data] == ["vllm_model"]
        assert models.data[0].owned_by == "Triton Inference Server"
        assert await client.is_ready() is True


@respx.mock
def test_sync_client_over_frontend() -> None:
    respx.post(f"{BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, json=_chat_payload("sync ok")))
    with OpenAICompatClient(settings=_settings()) as client:
        assert client.native is None
        response = client.chat.completions.create(model="vllm_model", messages=MESSAGES)
    assert response.choices[0].message.content == "sync ok"


async def test_loopback_frontend_disables_env_proxy() -> None:
    loopback = AsyncOpenAIFrontend("localhost:9000")
    remote = AsyncOpenAIFrontend("triton.example:9000")
    try:
        assert loopback._client.trust_env is False
        assert remote._client.trust_env is True
    finally:
        await loopback.close()
        await remote.close()


async def test_default_model_required_for_frontend() -> None:
    async with AsyncOpenAICompatClient(settings=_settings()) as client:
        with pytest.raises(ValueError, match="TRITON_DEFAULT_MODEL"):
            await client.chat.completions.create(messages=MESSAGES)
