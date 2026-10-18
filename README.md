# Triton-sdk

Python SDK for NVIDIA Triton Inference Server, focused on LLM workloads
(TensorRT-LLM and vLLM backends). One public API — `generate` / `stream` / `chat`,
with backend and transport (HTTP / gRPC) differences hidden inside.

```python
from triton_sdk import AsyncTritonLLMClient, SamplingParams

async with AsyncTritonLLMClient("localhost:8001", protocol="grpc") as client:
    async for chunk in client.stream("llama3", "Hello!", SamplingParams(max_tokens=64)):
        print(chunk.text, end="")
```

## OpenAI-compatible interface

Code written for `openai` is ported by swapping the constructor:

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
resp = client.chat.completions.create(model="vllm_model", messages=[{"role": "user", "content": "Hello"}])
resp.choices[0].message.content, resp.usage.total_tokens

for chunk in client.chat.completions.create(model="vllm_model", messages=[...], stream=True):
    chunk.choices[0].delta.content
```

Two engines behind one interface:

- `protocol="http" | "grpc"` — native KServe v2, works with any Triton; the chat template and `usage` are computed on the client.
- `protocol="openai"` — Triton's OpenAI frontend (`/v1/chat/completions`, port 9000); the template and `usage` are handled by the server,
`native` is unavailable.
`usage` is computed by the tokenizer from `tokenizer=` (without it — `None`). `tools`, `response_format`, `n > 1` are not supported
and raise `UnsupportedParameterError`; `extra_body={"top_k": ...}` forwards backend-specific parameters.



## Documentation

- [docs/usage.md](docs/usage.md) — **usage guide**: installation, choosing a client and protocol, configuration, streaming and cancellation, chat templates, errors, TLS, recipes
- [examples/README.md](examples/README.md) — how to talk to the model: curl, `openai`, our SDK
- [docs/architecture.md](docs/architecture.md) — how the SDK is built, layers, decisions
- [docs/backends.md](docs/backends.md) — vLLM / TensorRT-LLM schemas, parameter mapping table
- [monitoring/README.md](monitoring/README.md) — Prometheus + Grafana



## Structure


| Path                         | Role                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `triton_sdk/client/`         | Public API: `TritonLLMClient` (sync) and `AsyncTritonLLMClient`                                   |
| `triton_sdk/transport/`      | Network layer: `TransportInterface` + `HttpTransport` / `GrpcTransport`, SSE parser, protobuf stubs |
| `triton_sdk/backends/`       | Input/output schema adapters: `tensorrt_llm`, `vllm`, `generic` (generate endpoint)               |
| `triton_sdk/chat/`           | Chat templates: message list → raw prompt, token counting                                     |
| `triton_sdk/openai/`         | OpenAI-compatible facade: `chat.completions`, `completions`, `models` with types mirroring `openai`      |
| `triton_sdk/core/models/`    | Pydantic models for requests, responses, metadata, tensors                                            |
| `triton_sdk/core/constants/` | Protocols, KServe v2 paths, error codes, retry statuses                                              |
| `triton_sdk/settings.py`     | `TritonSettings` (`pydantic-settings`, `TRITON_` prefix)                                          |
| `triton_sdk/exceptions.py`   | `TritonError` hierarchy                                                                            |
| `triton_sdk/tests/`          | `unit/` (transport mocks), `integration/` (live Triton, `integration` marker)                     |
| `examples/`                  | Ready-made usage scenarios                                                                             |
| `model_repository/`          | Minimal vLLM model for integration tests                                                             |
| `monitoring/`                | Prometheus + Grafana (Triton LLM dashboard), see `monitoring/README.md`                              |




## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv sync --all-extras
cp triton_sdk/config/environment/env.sh.template env.sh && source env.sh
./code_quality_check.sh                                # ruff + mypy + pytest (unit)
```

Integration tests (needs a GPU and Docker with the NVIDIA Container Toolkit):

```bash
docker compose --profile openai up -d   # OpenAI frontend + KServe HTTP/gRPC in one container
scripts/wait_for_triton.sh
pytest -m integration
```

Monitoring (Prometheus + Grafana with a ready-made dashboard for Triton/vLLM):

```bash
docker compose --profile openai --profile monitoring up -d   # Grafana: http://localhost:3000
uv run python examples/load.py --concurrency 8 --requests 200
```

How to talk to the model (curl / `openai` / our SDK) — [examples/README.md](examples/README.md),
`uv run python examples/quickstart.py` runs through all paths.

gRPC stubs are generated from Triton's protobuf: `scripts/gen_protos.sh`.

## Key decisions

- Async core, sync client — a thin wrapper.
- gRPC is preferred for streaming (decoupled mode) and server-side generation cancellation.
- The backend is auto-detected from `model_config` (input names), or can be set explicitly.
- The library does not configure the root logger; `triton_sdk.config.configure_logging()` is opt-in for examples and tests.
