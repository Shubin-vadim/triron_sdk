"""Token streaming over gRPC with early cancellation. Run: uv run python examples/stream_async.py"""

import asyncio

from triton_sdk import AsyncTritonLLMClient, SamplingParams


async def main() -> None:
    async with AsyncTritonLLMClient("localhost:8001", protocol="grpc") as client:
        async for chunk in client.stream(
            "Write a long poem about GPUs.", SamplingParams(max_tokens=512), model="vllm_model"
        ):
            print(chunk.text, end="", flush=True)
            if chunk.index >= 50:  # noqa: PLR2004 - breaking out cancels generation on the server
                break
    print()


if __name__ == "__main__":
    asyncio.run(main())
