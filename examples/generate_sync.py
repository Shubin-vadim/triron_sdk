"""Blocking one-shot generation. Run: uv run python examples/generate_sync.py"""

from triton_sdk import SamplingParams, TritonLLMClient
from triton_sdk.config import configure_logging


def main() -> None:
    configure_logging("INFO")
    with TritonLLMClient("localhost:8000", protocol="http") as client:
        client.wait_for_ready("vllm_model", timeout=120)
        response = client.generate(
            "Explain Triton Inference Server in one sentence.", SamplingParams(max_tokens=64), model="vllm_model"
        )
        print(response.text)


if __name__ == "__main__":
    main()
