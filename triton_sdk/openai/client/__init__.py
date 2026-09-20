"""OpenAI-shaped facade over Triton.

    client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
    client.chat.completions.create(model="vllm_model", messages=[...], max_tokens=64)
    for chunk in client.chat.completions.create(..., stream=True): ...

Same attribute paths and return shapes as the openai SDK. Two engines behind the same surface:

- protocol="http" | "grpc"  -> KServe v2 through the native client (works with any Triton LLM deployment)
- protocol="openai"         -> Triton's OpenAI frontend on /v1/* (templating and usage done server-side)
"""

from triton_sdk.openai.client.async_client import AsyncOpenAICompatClient
from triton_sdk.openai.client.sync_client import OpenAICompatClient


__all__ = ["AsyncOpenAICompatClient", "OpenAICompatClient"]
