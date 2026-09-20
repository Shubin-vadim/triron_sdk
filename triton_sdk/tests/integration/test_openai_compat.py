"""OpenAI facade against live Triton: native engines (http/grpc) and the OpenAI frontend (compose profile openai)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import pytest

from triton_sdk.core.constants import Protocol
from triton_sdk.openai import AsyncOpenAICompatClient
from triton_sdk.settings import TritonSettings


pytestmark = pytest.mark.integration

MODEL = os.environ.get("TRITON_DEFAULT_MODEL", "vllm_model")
TOKENIZER = os.environ.get("TRITON_CHAT_TEMPLATE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HTTP_URL = os.environ.get("TRITON_URL", "localhost:8000")
GRPC_URL = os.environ.get("TRITON_GRPC_URL", "localhost:8001")
OPENAI_URL = os.environ.get("TRITON_OPENAI_URL", "localhost:9000")
MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Name one planet."},
]


def _frontend_available() -> bool:
    try:
        return httpx.get(f"http://{OPENAI_URL}/health/ready", timeout=2, trust_env=False).status_code == httpx.codes.OK
    except httpx.TransportError:
        return False


@pytest.fixture(params=["http", "grpc", "openai"])
async def client(request: pytest.FixtureRequest) -> AsyncIterator[AsyncOpenAICompatClient]:
    protocol = Protocol(request.param)
    if protocol is Protocol.OPENAI and not _frontend_available():
        pytest.skip("OpenAI frontend not running (docker compose --profile openai)")
    url = {Protocol.HTTP: HTTP_URL, Protocol.GRPC: GRPC_URL, Protocol.OPENAI: OPENAI_URL}[protocol]
    settings = TritonSettings(URL=url, PROTOCOL=protocol, DEFAULT_MODEL=MODEL, _env_file=None)  # type: ignore[call-arg]
    tokenizer = TOKENIZER if protocol is not Protocol.OPENAI else None
    async with AsyncOpenAICompatClient(settings=settings, tokenizer=tokenizer) as instance:
        yield instance


async def test_chat_completion(client: AsyncOpenAICompatClient) -> None:
    response = await client.chat.completions.create(messages=MESSAGES, max_tokens=16, temperature=0.0)
    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content
    assert response.choices[0].finish_reason in ("stop", "length")
    assert response.usage is not None
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


async def test_chat_stream(client: AsyncOpenAICompatClient) -> None:
    stream = await client.chat.completions.create(messages=MESSAGES, max_tokens=16, temperature=0.0, stream=True)
    chunks = [chunk async for chunk in stream]
    assert len({chunk.id for chunk in chunks}) == 1
    text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert text
    finals = [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices and chunk.choices[0].finish_reason]
    assert finals[-1] in ("stop", "length")


async def test_models(client: AsyncOpenAICompatClient) -> None:
    models = await client.models.list()
    assert MODEL in [model.id for model in models.data]
