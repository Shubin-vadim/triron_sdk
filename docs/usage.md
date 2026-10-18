# triton-sdk — usage guide

SDK for NVIDIA Triton Inference Server, focused on LLMs (vLLM and TensorRT-LLM backends). One public
API over HTTP, gRPC, and Triton's OpenAI frontend. Verified against Triton 25.08 + vllm_backend.

Contents:

1. [Installation](#1-installation)
2. [Which client to pick](#2-which-client-to-pick)
3. [Quick start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [`TritonLLMClient` — native client](#5-tritonllmclient--native-client)
6. [`OpenAICompatClient` — interface like `openai`](#6-openaicompatclient--interface-like-openai)
7. [Generation parameters](#7-generation-parameters)
8. [Streaming and cancellation](#8-streaming-and-cancellation)
9. [Chat templates](#9-chat-templates)
10. [Backends: vLLM and TensorRT-LLM](#10-backends-vllm-and-tensorrt-llm)
11. [Errors, timeouts, retries](#11-errors-timeouts-retries)
12. [TLS and authentication](#12-tls-and-authentication)
13. [Sync vs async](#13-sync-vs-async)
14. [Testing your own code](#14-testing-your-own-code)
15. [Logging](#15-logging)
16. [Recipes](#16-recipes)
17. [Known limitations](#17-known-limitations)

---

## 1. Installation

```bash
pip install triton-sdk              # HTTP transport, minimal dependencies (httpx, pydantic, tenacity)
pip install "triton-sdk[grpc]"      # + gRPC (grpcio, protobuf) — recommended for production
pip install "triton-sdk[chat]"      # + transformers for HF chat templates and usage counting
pip install "triton-sdk[all]"
```

Python ≥ 3.12. gRPC stubs are already in the package, `grpcio-tools` is not needed.

## 2. Which client to pick

| Task | Client |
|---|---|
| Porting code written for `openai` | `triton_sdk.openai.OpenAICompatClient` |
| Need full control: backend parameters, custom tensors, metadata, cancellation | `triton_sdk.TritonLLMClient` |
| Asynchronous service (FastAPI, aiohttp) | `AsyncTritonLLMClient` / `AsyncOpenAICompatClient` |

And which protocol:

| `protocol=` | Port | When |
|---|---|---|
| `"grpc"` | 8001 | **Default for production**: 2–3x lower overhead, server-side generation cancellation |
| `"http"` | 8000 | No `grpcio`, proxy/ingress is HTTP-only, debugging via curl |
| `"openai"` | 9000 | `OpenAICompatClient` only; Triton runs with the OpenAI frontend; template and `usage` computed by the server |

## 3. Quick start

```python
from triton_sdk import TritonLLMClient, SamplingParams

with TritonLLMClient("localhost:8001", protocol="grpc") as client:
    client.wait_for_ready("vllm_model", timeout=120)  # the model may take minutes to load

    response = client.generate(
        "The capital of France is",
        SamplingParams(max_tokens=32, temperature=0.0),
        model="vllm_model",
    )
    print(response.text)

    for chunk in client.stream("Tell me a joke", SamplingParams(max_tokens=128), model="vllm_model"):
        print(chunk.text, end="", flush=True)
```

Or in `openai` style:

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
r = client.chat.completions.create(
    model="vllm_model",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=64,
)
print(r.choices[0].message.content, r.usage.total_tokens)
```

The model name is the folder name in Triton's `model_repository`. If it's always the same, set
`TRITON_DEFAULT_MODEL` and you can skip passing `model=`.

## 4. Configuration

All settings are environment variables prefixed with `TRITON_` (or `.env`); any constructor parameter
overrides them. Full list — `triton_sdk/config/environment/env.sh.template`.

| Variable | Default | Meaning |
|---|---|---|
| `TRITON_URL` | `localhost:8000` | `host:port`; a `grpcs://` / `https://` scheme enables TLS |
| `TRITON_PROTOCOL` | `http` | `http` / `grpc` / `openai` |
| `TRITON_DEFAULT_MODEL` | — | model used if none is passed to the call |
| `TRITON_BACKEND` | auto | `vllm` / `tensorrt_llm` / `generic` — force the adapter choice |
| `TRITON_TIMEOUT` | `60` | unary request timeout, seconds |
| `TRITON_STREAM_TIMEOUT` | `300` | **pause between stream chunks**, seconds (not total duration) |
| `TRITON_CONNECT_TIMEOUT` | `5` | |
| `TRITON_RETRY_COUNT` / `TRITON_RETRY_START_TIMEOUT` | `3` / `1.0` | retries for non-streaming requests |
| `TRITON_TOKEN` | — | sent as `Authorization: Bearer …` |
| `TRITON_EXTRA_HEADERS` | `{}` | JSON dict of extra headers/metadata |
| `TRITON_CHAT_TEMPLATE_MODEL` | — | HF tokenizer id for the chat template |
| `TRITON_SSL*` | — | see [TLS](#12-tls-and-authentication) |

Programmatically — via `TritonSettings`:

```python
from triton_sdk import TritonLLMClient, TritonSettings

settings = TritonSettings(URL="triton.internal:8001", PROTOCOL="grpc", DEFAULT_MODEL="ensemble", TIMEOUT=30)
client = TritonLLMClient(settings=settings)
```

## 5. `TritonLLMClient` — native client

```python
TritonLLMClient(
    url: str | None = None,            # otherwise TRITON_URL
    *,
    protocol: "http" | "grpc" | None,  # otherwise TRITON_PROTOCOL
    backend: "vllm" | "tensorrt_llm" | "generic" | None,   # otherwise auto-detected from the model config
    settings: TritonSettings | None,
    chat_template: ChatTemplate | None,  # see section 9
    transport: TransportInterface | None,  # for tests: swap out the network
)
```

Methods (`AsyncTritonLLMClient` has the same ones, but `await`-ed):

| Method | What it does |
|---|---|
| `is_ready(model=None)` | `True` if the server (or model) is ready |
| `wait_for_ready(model=None, timeout=300, interval=2)` | Polls readiness; `ModelNotReadyError` on timeout |
| `model_metadata(model)` / `model_config(model)` | Model metadata and `config.pbtxt` (cached) |
| `list_models()` | Repository index: name, version, state |
| `generate(prompt, params=None, *, model=None, model_version=None, request_id=None, timeout=None, extra_inputs=None, exclude_input_in_output=True)` → `GenerateResponse` | A single, complete response |
| `stream(prompt, params=None, *, model=None, model_version=None, request_id=None, timeout=None, extra_inputs=None, exclude_input_in_output=True)` → iterator of `GenerateChunk` | Tokens as they are generated |
| `chat(messages, params=None, **kw)` / `chat_stream(messages, params=None, **kw)` | Same, but the input is a list of `ChatMessage`, with the template applied on the client |
| `render_chat(messages)` → `str` | Preview the prompt after the template is applied |
| `close()` / context manager | Close connections |

Results:

```python
response.text  # str
response.finish_reason  # FinishReason.STOP / LENGTH / UNKNOWN (Triton backends don't report the reason)
response.cum_log_probs  # float | None, if log-probs were requested (TRT-LLM)
response.raw  # raw backend response, for debugging

chunk.text  # text delta
chunk.is_final  # last chunk of the stream
chunk.log_probs  # list[float] | None
```

Important note on decoupled models: Triton's LLM models are almost always decoupled (streaming). For these,
`generate()` **automatically** goes through the streaming API with `stream=false` and assembles the response —
Triton rejects a plain unary `infer` for such models. You don't need to do anything, but you'll see
`ModelStreamInfer` in the gRPC trace.

## 6. `OpenAICompatClient` — interface like `openai`

The same attribute paths and response shapes as the `openai` package (no dependency on it):

```python
from triton_sdk.openai import OpenAICompatClient, AsyncOpenAICompatClient

client = OpenAICompatClient(
    "localhost:8001",
    protocol="grpc",  # "http" | "grpc" — native path; "openai" — Triton's frontend
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct",  # HF chat template + usage counting (needs the [chat] extra)
)

# chat
r = client.chat.completions.create(model="vllm_model", messages=[...], max_tokens=64, temperature=0.2, stop=["\n\n"])
r.id, r.model, r.choices[0].message.content, r.choices[0].finish_reason, r.usage.prompt_tokens

# streaming — the same chunk sequence as OpenAI: role → deltas → finish_reason → (usage)
for chunk in client.chat.completions.create(
    model="vllm_model", messages=[...], stream=True, stream_options={"include_usage": True}
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
    if chunk.usage:
        print(chunk.usage)

# legacy completions
c = client.completions.create(model="vllm_model", prompt="Once upon a time", max_tokens=32)
c.choices[0].text

# models
[m.id for m in client.models.list().data]

# access to the native client, if you need something beyond the openai interface
client.native.model_config("vllm_model")  # None when protocol="openai"
```

Supported `create` arguments: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `seed`,
`stop` (str or list), `presence_penalty`, `frequency_penalty`, `logprobs`, `stream`, `stream_options`,
`extra_body` (backend-specific: `{"top_k": 5, "repetition_penalty": 1.2, "min_tokens": 4, "ignore_eos": True}`).

Ignored without error: `user`, `metadata`, `store`, `service_tier`, `timeout`, `extra_headers`.
Raise `UnsupportedParameterError`: `tools`, `tool_choice`, `functions`, `response_format`, `logit_bias`,
`audio`, `n > 1`. Typos in argument names → `TypeError`.

Differences between the two engines:

| | `protocol="http" / "grpc"` | `protocol="openai"` |
|---|---|---|
| Chat template | on the client (`tokenizer=` or `chat_template=`) | on the server (`--tokenizer` at frontend startup) |
| `usage` | computed by the client's tokenizer, `None` without `tokenizer=`; can differ from the server by ±1 token due to special tokens | exact, from the server |
| `finish_reason` | `"length"` if `completion_tokens >= max_tokens`, otherwise `"stop"` | from the server |
| `native` | available | `None` |
| Works without the OpenAI frontend | yes | no |

## 7. Generation parameters

`SamplingParams` — a single set for all backends; `None` means "don't send", the model's default applies.

```python
SamplingParams(
    max_tokens=256,  # required budget
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=None,
    presence_penalty=None,
    frequency_penalty=None,
    length_penalty=None,
    min_tokens=None,
    beam_width=None,
    seed=None,
    stop=[],
    bad_words=[],
    return_log_probs=False,  # TRT-LLM: cum_log_probs + output_log_probs; vLLM: ignored
    ignore_eos=False,  # vLLM: native; TRT-LLM: via end_id=-1
)
```

How a field turns into a tensor for a specific backend — see the table in [backends.md](backends.md). Parameters
that a model doesn't declare in its config are **not sent** (a DEBUG log is emitted), rather than breaking the request.

In addition to `params`, `generate()` / `stream()` accept `model_version=`, `request_id=` (for tracing),
`extra_inputs=` (custom tensors, see recipes), and `exclude_input_in_output=` (defaults to `True` — the prompt is
not echoed back in the response).

## 8. Streaming and cancellation

```python
async for chunk in client.stream(prompt, params, model="vllm_model"):
    print(chunk.text, end="")
    if enough:
        break  # ← this is enough
```

- `break` (or closing the generator) **cancels the request on the server**: over gRPC `call.cancel()` is called,
  and Triton stops generating; over HTTP the connection is torn down (the vLLM backend notices this too).
- The last chunk is always marked `is_final=True`, even if the server (HTTP `generate_stream`) didn't send a marker.
- `timeout` on `stream()` is the maximum pause between two chunks, not the total duration.
- A stream is **never retried**; `generate()` is retried on network errors and 5xx/429 (`RETRY_COUNT`).
- In the sync client, `for chunk in client.stream(...)` works the same way; breaking out of the loop closes the stream.

## 9. Chat templates

Triton's vLLM and TRT-LLM backends accept **raw text** — the chat template must be applied on the client
(except for `protocol="openai"`, where the frontend does it).

```python
from triton_sdk.chat import ChatMessage, Role, HFChatTemplate, SimpleChatTemplate

template = HFChatTemplate("Qwen/Qwen2.5-0.5B-Instruct")  # tokenizer.apply_chat_template, needs the [chat] extra
client = TritonLLMClient("localhost:8001", protocol="grpc", chat_template=template)

messages = [
    ChatMessage(role=Role.SYSTEM, content="Answer concisely."),
    ChatMessage(role=Role.USER, content="What is Triton?"),
]
client.chat(messages, SamplingParams(max_tokens=64), model="vllm_model")
```

- Without `chat_template=`, `SimpleChatTemplate` is used (`role: content` line by line) — fine for tests, but
  instruct models will respond worse and tend to "continue" the prompt. Always set an HF template for real models.
- A custom template is any object with `render(messages, *, add_generation_prompt=True) -> str` and
  `count_tokens(text) -> int | None` (for `usage`).

## 10. Backends: vLLM and TensorRT-LLM

The adapter is chosen automatically from the model's `config.pbtxt` (input names), or forced via `TRITON_BACKEND`.
The adapter knows the model schema: it builds the shape to match `max_batch_size`, sends only declared inputs,
and resolves renames across versions (`min_length` → `min_tokens`).

| | vLLM (`vllm_backend`) | TensorRT-LLM (`ensemble` / `tensorrt_llm_bls`) |
|---|---|---|
| Detection | input `sampling_parameters` or `backend: vllm` | inputs `text_input` + `max_tokens` |
| Parameters | JSON string `sampling_parameters` | one tensor per parameter |
| Streaming | deltas | deltas |
| Status in the SDK | **verified against a live Triton 25.08** | covered by unit tests, a live run on your deployment is required |

Details and the parameter mapping table — [backends.md](backends.md).

## 11. Errors, timeouts, retries

Only subclasses of `triton_sdk.TritonError` (`.message`, `.status_code`, `.model`) are ever raised outward:

| Exception | When |
|---|---|
| `TritonConnectionError` | server unreachable, connection dropped, TLS handshake failed |
| `TritonTimeoutError` | `TIMEOUT` exceeded / stream pause > `STREAM_TIMEOUT` |
| `ModelNotFoundError` | unknown model/version (404 / `NOT_FOUND`) |
| `ModelNotReadyError` | model is loading; `wait_for_ready` timed out |
| `InferenceError` | backend rejected or failed to execute the request (including an error mid-stream) |
| `StreamCancelledError` | stream was cancelled (gRPC `CANCELLED`) |
| `BackendNotSupportedError` | the model config didn't match any adapter |
| `MissingDependencyError` | an extra is needed: `pip install "triton-sdk[grpc]"` / `[chat]` |
| `openai.UnsupportedParameterError` | an openai argument that can't be expressed on Triton |

```python
from triton_sdk import TritonError
from triton_sdk.exceptions import ModelNotReadyError, TritonTimeoutError

try:
    client.generate(prompt, model="vllm_model")
except ModelNotReadyError:
    client.wait_for_ready("vllm_model")
except TritonTimeoutError as exc:
    log.warning("slow model %s: %s", exc.model, exc.message)
except TritonError as exc:
    log.error("triton %s: %s", exc.status_code, exc.message)
```

Retries: `generate()` retries the request on `TritonConnectionError` and statuses 408/429/500/502/503/504 with
exponential backoff (`RETRY_COUNT`, `RETRY_START_TIMEOUT`). Streams are never retried.

## 12. TLS and authentication

```bash
export TRITON_URL=grpcs://triton.internal:8001     # scheme enables TLS; or TRITON_SSL=true
export TRITON_SSL_ROOT_CERT=/etc/ssl/corp-ca.pem    # custom CA; empty = system roots
export TRITON_SSL_CLIENT_CERT=/etc/ssl/client.pem   # mTLS
export TRITON_SSL_CLIENT_KEY=/etc/ssl/client.key
export TRITON_SSL_SERVER_NAME_OVERRIDE=triton       # if the certificate isn't issued for this host
export TRITON_TOKEN=...                             # Authorization: Bearer
```

The same settings apply to HTTP, gRPC, and the OpenAI frontend. `TRITON_VERIFY_SSL=false` disables certificate
verification **for HTTP only** — gRPC always verifies the server (the SDK logs a warning about it).

## 13. Sync vs async

- `AsyncTritonLLMClient` / `AsyncOpenAICompatClient` — the core; use these in asyncio applications.
- `TritonLLMClient` / `OpenAICompatClient` — wrappers that run their own event loop in a background thread.
  Safe to call from regular code, but **do not** call them from a coroutine (you'll block the loop).
- One client = one connection pool; create it once per process, not per request. Clients are
  thread-safe for concurrent calls.

```python
# FastAPI
app.state.llm = AsyncTritonLLMClient("triton:8001", protocol="grpc")


@app.post("/complete")
async def complete(req: Req):
    r = await app.state.llm.generate(req.prompt, SamplingParams(max_tokens=req.max_tokens))
    return {"text": r.text}
```

## 14. Testing your own code

Swap out the transport — no network needed:

```python
from triton_sdk import AsyncTritonLLMClient
from triton_sdk.core.models import DataType, InferOutput, InferResult, ModelConfig
from triton_sdk.transport import TransportInterface


class FakeTransport(TransportInterface):
    async def model_config(self, model, version=None):
        return ModelConfig(
            name=model,
            backend="vllm",
            max_batch_size=0,
            model_transaction_policy={"decoupled": True},
            input=[{"name": "text_input"}, {"name": "sampling_parameters"}],
        )

    async def stream_infer(self, model, inputs, **kw):
        yield InferResult(
            model_name=model,
            outputs={
                "text_output": InferOutput(name="text_output", datatype=DataType.BYTES, shape=[1], data=["stubbed"])
            },
            parameters={"triton_final_response": True},
        )

    # the remaining interface methods — as needed


client = AsyncTritonLLMClient(transport=FakeTransport())
```

A ready-made `FakeTransport` that records calls is in `triton_sdk/tests/conftest.py` — feel free to copy it.

## 15. Logging

The SDK writes to `triton_sdk.*` loggers and **does not touch** the root logger. For debugging:

```python
import logging

logging.getLogger("triton_sdk").setLevel(logging.DEBUG)  # adapter selection, dropped parameters, cancellations
```

`triton_sdk.config.configure_logging("DEBUG")` is a convenient opt-in for scripts (colored output, quiets httpx/grpc).

## 16. Recipes

**Custom ensemble / extra tensors** — `extra_inputs=`; the type is taken from the model's `config.pbtxt`, for
known TRT-LLM inputs (`end_id`, `pad_id`, `embedding_bias_*`, `lora_task_id`, …) — from the adapter's table, otherwise
guessed from the value. A ready-made `InferInput` is passed through as-is:

```python
from triton_sdk.core.models import DataType, InferInput

client.generate(
    "...",
    SamplingParams(max_tokens=64),
    model="ensemble",
    extra_inputs={
        "end_id": 2,
        "embedding_bias_words": ["foo", "bar"],
        "custom": InferInput(name="custom", datatype=DataType.FP16, shape=[1, 1], data=[0.5]),
    },
)
```

**Multiple models on one server** — a single client, `model=` on each call; configs are cached per name.

**Choosing a model version** — `client.generate(..., model_version="2")`; without a version Triton picks the
latest according to the model's `version_policy`.

**Production profile** — gRPC + TLS + `DEFAULT_MODEL` + `STREAM_TIMEOUT` tuned for your response lengths +
`RETRY_COUNT=3`; create the client once at process startup; use `wait_for_ready()` in your service's readiness probe.

**Load / benchmarking** — `examples/load.py --concurrency N --requests M`, metrics in Grafana
(`monitoring/README.md`).

## 17. Known limitations

- `n > 1`, `tools` / function calling, `response_format` (JSON mode), `logit_bias` — not supported
  by Triton backends through this SDK; raises `UnsupportedParameterError`.
- `usage` on the native path is an estimate from the client's tokenizer (±1–2 tokens), or `None` without `tokenizer=`.
- `finish_reason` isn't reported by the backends; `"length"` is derived from the token budget.
- vLLM's gauge metrics (KV-cache, scheduler queue) aren't exposed through Triton — only counters/histograms.
- The TensorRT-LLM adapter hasn't been verified against a live server (see section 10).
- Multimodal inputs (`image`) are only supported via `extra_inputs`; in the openai facade, content parts other
  than `text` are rejected.
