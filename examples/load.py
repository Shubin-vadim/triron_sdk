"""Generate load so the Grafana dashboard has something to show.

Run: uv run python examples/load.py --concurrency 8 --requests 100 [--protocol grpc] [--url localhost:8001]
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time

from triton_sdk import AsyncTritonLLMClient, SamplingParams


PROMPTS = [
    "Explain what a GPU is in two sentences.",
    "Write a haiku about inference servers.",
    "List three uses of Python.",
    "What is the capital of Japan?",
    "Describe the water cycle briefly.",
    "Why is the sky blue?",
]


async def worker(client: AsyncTritonLLMClient, model: str, queue: asyncio.Queue[int], stats: dict[str, float]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        prompt = random.choice(PROMPTS)  # noqa: S311 - not security related
        params = SamplingParams(max_tokens=random.randint(32, 128), temperature=0.7)  # noqa: S311
        t0 = time.perf_counter()
        tokens = 0
        async for _ in client.stream(prompt, params, model=model):
            tokens += 1
        stats["requests"] += 1
        stats["chunks"] += tokens
        stats["seconds"] += time.perf_counter() - t0


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8001")
    parser.add_argument("--protocol", default="grpc", choices=["http", "grpc"])
    parser.add_argument("--model", default="vllm_model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(args.requests):
        queue.put_nowait(i)
    stats = {"requests": 0.0, "chunks": 0.0, "seconds": 0.0}

    started = time.perf_counter()
    async with AsyncTritonLLMClient(args.url, protocol=args.protocol) as client:
        await client.wait_for_ready(args.model, timeout=120)
        await asyncio.gather(*(worker(client, args.model, queue, stats) for _ in range(args.concurrency)))
    wall = time.perf_counter() - started

    print(
        f"{int(stats['requests'])} requests, {int(stats['chunks'])} chunks in {wall:.1f}s "
        f"-> {stats['requests'] / wall:.2f} req/s, {stats['chunks'] / wall:.1f} chunks/s, "
        f"avg latency {stats['seconds'] / max(stats['requests'], 1):.2f}s at concurrency {args.concurrency}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # model output is unicode; Windows consoles default to cp1252
    asyncio.run(main())
