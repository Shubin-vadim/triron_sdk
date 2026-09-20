"""Live tests against docker-compose Triton (vLLM backend). Run: pytest -m integration."""

from collections.abc import AsyncIterator

import pytest

from triton_sdk import AsyncTritonLLMClient, SamplingParams


pytestmark = pytest.mark.integration


@pytest.fixture
async def client() -> AsyncIterator[AsyncTritonLLMClient]:
    async with AsyncTritonLLMClient() as instance:  # URL / model from TRITON_* env
        await instance.wait_for_ready("vllm_model", timeout=600)
        yield instance


async def test_generate_returns_text(client: AsyncTritonLLMClient) -> None:
    response = await client.generate("Say hello.", SamplingParams(max_tokens=8, temperature=0.0), model="vllm_model")
    assert response.text.strip()


async def test_stream_emits_final_chunk(client: AsyncTritonLLMClient) -> None:
    chunks = [
        chunk async for chunk in client.stream("Count to three.", SamplingParams(max_tokens=16), model="vllm_model")
    ]
    assert chunks
    assert chunks[-1].is_final
