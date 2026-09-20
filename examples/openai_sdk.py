"""Smoke test through the official `openai` package against Triton's OpenAI frontend.

Run (after `docker compose --profile openai up -d`):
    uv run --with openai python examples/openai_sdk.py
"""

from __future__ import annotations

import sys
import time

import httpx


try:
    from openai import APIStatusError, OpenAI
except ImportError as exc:  # pragma: no cover - example script
    msg = "Install the openai extra: uv run --with openai python examples/openai_sdk.py"
    raise SystemExit(msg) from exc


BASE_URL = "http://localhost:9000/v1"
MODEL = "vllm_model"
READY_TIMEOUT = 120.0


def wait_for_frontend(client: OpenAI, timeout: float = READY_TIMEOUT) -> list[str]:
    """Poll GET /v1/models until the frontend is up; 503 is Docker/proxy still binding port 9000."""
    deadline = time.monotonic() + timeout
    last: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return [model.id for model in client.models.list().data]
        except (httpx.TransportError, APIStatusError) as exc:
            last = exc
            time.sleep(2)
    msg = f"OpenAI frontend not ready at {BASE_URL}"
    raise RuntimeError(msg) from last


def main() -> None:
    # trust_env=False: do not send localhost through HTTP_PROXY (the proxy returns 503, Triton never sees it).
    http_client = httpx.Client(trust_env=False, timeout=120.0)
    client = OpenAI(base_url=BASE_URL, api_key="unused", http_client=http_client)

    models = wait_for_frontend(client)
    print("models:", models)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is Triton Inference Server? Answer in one sentence."},
        ],
        max_tokens=64,
        temperature=0.2,
    )
    print("answer:", response.choices[0].message.content)
    print("finish_reason:", response.choices[0].finish_reason)
    print("usage:", response.usage)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
