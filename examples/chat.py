"""Chat with a Hugging Face chat template applied client-side. Needs the chat extra."""

import asyncio

from triton_sdk import AsyncTritonLLMClient, SamplingParams
from triton_sdk.chat import ChatMessage, HFChatTemplate, Role


async def main() -> None:
    template = HFChatTemplate("Qwen/Qwen2.5-0.5B-Instruct")
    async with AsyncTritonLLMClient("localhost:8000", chat_template=template) as client:
        messages = [
            ChatMessage(role=Role.SYSTEM, content="You are a concise assistant."),
            ChatMessage(role=Role.USER, content="What is KServe v2?"),
        ]
        response = await client.chat(messages, SamplingParams(max_tokens=128), model="vllm_model")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
