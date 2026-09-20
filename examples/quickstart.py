"""Call the same model every way the SDK supports and print answers + timings.

Run: uv run python examples/quickstart.py   (expects docker compose --profile openai up -d)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from triton_sdk import AsyncTritonLLMClient, SamplingParams
from triton_sdk.chat import ChatMessage, Role
from triton_sdk.openai import AsyncOpenAICompatClient


MODEL = os.environ.get("TRITON_DEFAULT_MODEL", "vllm_model")
TOKENIZER = os.environ.get("TRITON_CHAT_TEMPLATE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HTTP, GRPC, OPENAI = "localhost:8000", "localhost:8001", "localhost:9000"
QUESTION = "Name one planet of the Solar System. Answer with one word."
PARAMS = SamplingParams(max_tokens=16, temperature=0.0)


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


async def native(protocol: str, url: str) -> None:
    banner(f"TritonLLMClient  protocol={protocol}  {url}")
    async with AsyncTritonLLMClient(url, protocol=protocol) as client:
        config = await client.model_config(MODEL)
        print(f"model={config.name} backend={config.backend} decoupled={config.decoupled}")

        # raw prompt, no chat template
        t0 = time.perf_counter()
        response = await client.generate("The capital of France is", PARAMS, model=MODEL)
        print(f"generate  {time.perf_counter() - t0:.2f}s -> {response.text!r}")

        # streaming with per-token timing; break early to demonstrate cancellation
        t0 = time.perf_counter()
        first: float | None = None
        pieces: list[str] = []
        async for chunk in client.stream("Count from one to twenty:", SamplingParams(max_tokens=64), model=MODEL):
            first = first or time.perf_counter() - t0
            pieces.append(chunk.text)
            if len(pieces) >= 12:  # noqa: PLR2004 - stop early: the transport cancels the request server-side
                break
        print(f"stream    ttft={first:.2f}s, {len(pieces)} chunks then cancelled -> {''.join(pieces)!r}")

        # chat: SDK applies the template client-side (SimpleChatTemplate here; pass chat_template= for HF)
        messages = [ChatMessage(role=Role.USER, content=QUESTION)]
        response = await client.chat(messages, PARAMS, model=MODEL)
        print(f"chat      -> {response.text!r}")


async def openai_compat(protocol: str, url: str) -> None:
    banner(f"OpenAICompatClient  protocol={protocol}  {url}")
    tokenizer = TOKENIZER if protocol != "openai" else None  # the frontend applies the template server-side
    async with AsyncOpenAICompatClient(url, protocol=protocol, tokenizer=tokenizer) as client:
        models = await client.models.list()
        print("models:", [model.id for model in models.data])

        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": QUESTION}], max_tokens=16, temperature=0.0
        )
        usage = response.usage
        print(
            f"chat.completions  {time.perf_counter() - t0:.2f}s -> {response.choices[0].message.content!r} "
            f"finish={response.choices[0].finish_reason} tokens={usage.total_tokens if usage else None}"
        )

        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello in three languages."}],
            max_tokens=32,
            stream=True,
        )
        pieces = [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
        print(f"stream            -> {''.join(pieces)!r}")


async def main() -> None:
    await native("http", HTTP)
    await native("grpc", GRPC)
    await openai_compat("grpc", GRPC)
    await openai_compat("openai", OPENAI)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # model output is unicode; Windows consoles default to cp1252
    asyncio.run(main())
