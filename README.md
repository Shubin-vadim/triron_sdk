# Triton-sdk

Python SDK for NVIDIA Triton Inference Server, focused on LLM workloads
(TensorRT-LLM and vLLM backends). One public API — `generate` / `stream` / `chat`,
with backend and transport (HTTP / gRPC) differences hidden inside.

Key features:

- **One LLM API across supported backends:** `generate`, `stream`, and `chat`
  adapt vLLM and TensorRT-LLM model schemas, including backend-specific inputs
  and outputs.
- **OpenAI-style interface for native Triton endpoints:** use
  `chat.completions` and `completions` over HTTP or gRPC without deploying
  Triton's OpenAI frontend. Compatibility is partial; see the limitations below.

## Quick start

Clone the repository:

```bash
git clone https://github.com/Shubin-vadim/triron_sdk.git
cd triron_sdk
```

Python 3.12 or newer is required. If `uv` is not installed, install it with pip:

```bash
python -m pip install uv
```

Install the SDK, optional dependencies, and development dependencies in one command:

```bash
uv sync --all-extras --dev
```

This installs the `triton-inference-sdk` distribution into the project's `.venv`; import it
in Python as `triton_sdk`. It does **not** install Triton Inference Server or vLLM: the
SDK is a client, and it sends requests to a Triton server where a model is already
running. You do not need Triton or vLLM installed on the computer running the SDK.

### Prepare Triton and an LLM model

If you do not already have a Triton deployment, follow NVIDIA's
[Triton Server Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)
to install and start the server. A running Triton process alone is not enough: it must
also have an LLM backend installed and a compatible model configured and loaded in its
model repository. The SDK cannot install or load these server-side components for you.

Choose the backend you plan to use and follow its setup guide:

- **vLLM:** NVIDIA's [Triton vLLM backend guide](https://github.com/triton-inference-server/vllm_backend)
  covers the backend-enabled container and sample model repository.
- **TensorRT-LLM:** NVIDIA's [TensorRT-LLM backend Getting Started guide](https://github.com/triton-inference-server/tensorrtllm_backend#getting-started)
  covers the compatible container, model configuration, and server launch. Match the
  container and NVIDIA driver to your GPU; gated Hugging Face models may require a token.

After following the backend guide, confirm the model is loaded and ready in Triton.
Then use its server address and Triton model name in the example below. The model's input
schema must be supported by this SDK.

### Connect to your Triton server

Once a Triton server is prepared, connect to it and pass the **Triton model name**
configured on that server. This can be a vLLM model or a TensorRT-LLM ensemble/BLS model
with a supported text-input schema; it is not necessarily the Hugging Face model ID.

```python
from triton_sdk import SamplingParams, TritonLLMClient

with TritonLLMClient("your-triton-host:8001", protocol="grpc") as client:
    response = client.generate(
        "The capital of France is",
        SamplingParams(max_tokens=32),
        model="your-triton-model-name",
    )
    print(response.text)

    for chunk in client.stream(
        "Tell me a short story.",
        SamplingParams(max_tokens=64),
        model="your-triton-model-name",
    ):
        print(chunk.text, end="", flush=True)
```

To run the repository's end-to-end example, set `TRITON_DEFAULT_MODEL` to your Triton
model name and run:

PowerShell:

```powershell
$env:TRITON_DEFAULT_MODEL = "your-triton-model-name"
uv run python examples/quickstart.py
```

On Linux/macOS:

```bash
export TRITON_DEFAULT_MODEL=your-triton-model-name
uv run python examples/quickstart.py
```

The example uses HTTP at `localhost:8000` and gRPC at `localhost:8001` by default; set
`TRITON_URL` and `TRITON_GRPC_URL` to override them. Set `TRITON_OPENAI_URL` to the
address of an OpenAI-compatible Triton frontend to exercise that protocol too; otherwise,
that part of the example is skipped. The example requires both native HTTP and gRPC
endpoints, and a model configured for streaming (HTTP streaming also requires Triton's
generate extension). It uses the Qwen2.5-0.5B-Instruct tokenizer by default for chat;
set `TRITON_CHAT_TEMPLATE_MODEL` to a tokenizer matching your model. The first run may
download the tokenizer from Hugging Face.
The SDK reads model configuration from the server to select an adapter.

The SDK does not install or launch Triton, backends, or models.

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

To run the standalone example against an already-running OpenAI-compatible Triton frontend
(defaults to `localhost:9000`, model `vllm_model`):

```bash
uv run --with openai python examples/openai_sdk.py
```

Set `TRITON_OPENAI_URL` and `TRITON_DEFAULT_MODEL` first if your frontend or model uses a
different address or name.

## Structure

| Path                           | Role                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `triton_sdk/client/`         | Public API: `TritonLLMClient` (sync) and `AsyncTritonLLMClient`                                         |
| `triton_sdk/transport/`      | Network layer: `TransportInterface` + `HttpTransport` / `GrpcTransport`, SSE parser, protobuf stubs   |
| `triton_sdk/backends/`       | Input/output schema adapters: `tensorrt_llm`, `vllm`, `generic` (generate endpoint)                   |
| `triton_sdk/chat/`           | Chat templates: message list → raw prompt, token counting                                                  |
| `triton_sdk/openai/`         | OpenAI-compatible facade: `chat.completions`, `completions`, `models` with types mirroring `openai` |
| `triton_sdk/core/models/`    | Pydantic models for requests, responses, metadata, tensors                                                  |
| `triton_sdk/core/constants/` | Protocols, KServe v2 paths, error codes, retry statuses                                                     |
| `triton_sdk/settings.py`     | `TritonSettings` (`pydantic-settings`, `TRITON_` prefix)                                              |
| `triton_sdk/exceptions.py`   | `TritonError` hierarchy                                                                                   |
| `triton_sdk/tests/`          | `unit/` (transport mocks), `integration/` (live Triton, `integration` marker)                         |
| `examples/`                  | Ready-made usage scenarios                                                                                  |
| `model_repository/`          | Example vLLM model configuration for a Triton deployment                                                     |

## Local development and integration tests

```bash
uv sync --all-extras --dev
cp triton_sdk/config/environment/env.sh.template env.sh && source env.sh
./code_quality_check.sh                                # ruff + mypy + pytest (unit)
```

Integration tests require a separately running Triton server and a vLLM model named
`vllm_model`, with native HTTP and gRPC endpoints. Configure the HTTP server address in
`env.sh`, then run `pytest -m integration`. The template sets the model name to
`vllm_model`; change it if your server uses another name. Set `TRITON_GRPC_URL` and
`TRITON_OPENAI_URL` when those endpoints use non-default addresses. Tests for Triton's
OpenAI frontend are skipped when the frontend is unavailable.

## License

This project is licensed under the [MIT License](LICENSE).
