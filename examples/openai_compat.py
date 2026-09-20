"""Drop-in replacement for code written against the openai SDK. Needs the chat extra for the tokenizer."""

from triton_sdk.openai import OpenAICompatClient


def main() -> None:
    client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "What is Triton Inference Server?"},
    ]

    response = client.chat.completions.create(model="vllm_model", messages=messages, max_tokens=64, temperature=0.2)
    print(response.choices[0].message.content)
    print(response.usage)

    stream = client.chat.completions.create(model="vllm_model", messages=messages, max_tokens=64, stream=True)
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()

    print([model.id for model in client.models.list().data])
    client.close()


if __name__ == "__main__":
    main()
