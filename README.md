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

This installs `triton-sdk` into the project's `.venv` on your computer. It does **not**
install Triton Inference Server or vLLM: the SDK is a client, and it sends requests to a
Triton server where a model is already running. You do not need Triton or vLLM installed
on the computer running the SDK.

### Start a local Triton server

The repository includes a Docker Compose setup for Triton with the vLLM backend and a
sample Qwen2.5-0.5B-Instruct model. The server and model run in Docker; they are not
installed into the SDK's Python environment.

You need Docker with the Compose plugin and an NVIDIA GPU accessible to Docker. On Linux,
install the NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
For Windows, use Docker Desktop with GPU support through WSL2. Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

From the repository directory, start Triton:

```bash
docker compose --profile openai up -d
docker compose logs -f openai
```

The first start downloads the model weights and may take a while. Wait until the logs show
that `vllm_model` is ready, then press Ctrl+C to stop following the logs (the server keeps
running). In a second terminal, run the SDK example:

```bash
uv run python examples/quickstart.py
```

The sample model is public, so no Hugging Face token is needed. The model name used by the
SDK is `vllm_model`. The server exposes HTTP on port 8000, gRPC on port 8001, and its
OpenAI frontend on port 9000. Stop it when done:

```bash
docker compose --profile openai down
```

To run Triton without the OpenAI frontend, use `docker compose --profile native up -d`
instead. The included Compose setup covers vLLM only. A TensorRT-LLM server requires its
own Triton deployment and model repository; see the
[Triton TensorRT-LLM backend documentation](https://github.com/triton-inference-server/tensorrtllm_backend).

### Connect to an existing Triton server instead

Connect to an existing Triton server and pass the **Triton model name** configured
on that server. This can be a vLLM model or a TensorRT-LLM ensemble/BLS model with a
supported text-input schema; it is not necessarily the Hugging Face model ID.

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

Save the code above as `quickstart.py`, replace the example server address and model name,
then run it:

```bash
uv run python quickstart.py
```

Use the server's HTTP port and `protocol="http"` for HTTP. Streaming requires a model
configured for streaming; HTTP streaming also requires Triton's generate extension.
The SDK reads model configuration from the server to select an adapter. For chat, use
the model's appropriate chat template.

The included `model_repository/` and Docker Compose setup are optional; they are not needed
when connecting to your own server. The SDK does not create or deploy models on Triton.

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
| `model_repository/`          | Optional local Triton model configuration for examples and integration tests                                |

## Local development and test server

The repository includes a `vllm_model` configuration for Qwen2.5-0.5B-Instruct. It contains
configuration files, not model weights. Docker Compose mounts it into the test server; the
vLLM backend downloads the weights to the `hf-cache` volume if they are not already present.
This local setup is separate from any model repository on your own Triton server.

```bash
uv sync --all-extras --dev
cp triton_sdk/config/environment/env.sh.template env.sh && source env.sh
./code_quality_check.sh                                # ruff + mypy + pytest (unit)
```

## License

This project is licensed under the [MIT License](LICENSE).
