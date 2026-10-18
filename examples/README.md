# How to talk to the model

All examples assume `docker compose --profile openai up -d` — one container serves three ports:

| Port | What | For whom |
|---|---|---|
| **9000** | OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`) | any OpenAI client, `curl`, our `OpenAICompatClient(protocol="openai")` |
| **8000** | KServe v2 HTTP (`/v2/models/<m>/generate`, `/infer`, health) | `curl`, our `TritonLLMClient(protocol="http")` |
| **8001** | KServe v2 gRPC (`ModelInfer`, `ModelStreamInfer`) | our `TritonLLMClient(protocol="grpc")` — the fastest path, with cancellation |

The model is named after its folder in `model_repository/` — here, `vllm_model`.

## 0. No code: curl

```bash
# OpenAI API — the server applies the chat template
curl -s localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "vllm_model",
  "messages": [{"role": "user", "content": "Name one planet."}],
  "max_tokens": 32
}'

# KServe generate — raw prompt, no template; vLLM parameters as a JSON string
curl -s localhost:8000/v2/models/vllm_model/generate -H 'Content-Type: application/json' -d '{
  "text_input": "The capital of France is",
  "sampling_parameters": "{\"max_tokens\": 8, \"temperature\": 0}",
  "exclude_input_in_output": true
}'

# same with streaming (SSE)
curl -sN localhost:8000/v2/models/vllm_model/generate_stream -H 'Content-Type: application/json' -d '{
  "text_input": "Counting: one, two,", "stream": true,
  "sampling_parameters": "{\"max_tokens\": 6, \"temperature\": 0}"
}'

# health and model list
curl -s localhost:8000/v2/health/ready -o /dev/null -w '%{http_code}\n'
curl -s localhost:9000/v1/models
```

## 1. The official `openai` SDK (none of ours needed)

[openai_sdk.py](openai_sdk.py) — `uv run --with openai python examples/openai_sdk.py`.

```python
import httpx
from openai import OpenAI

# trust_env=False: otherwise HTTP_PROXY might intercept localhost and return 503, and Triton never sees the request
client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="unused",
    http_client=httpx.Client(trust_env=False),
)
r = client.chat.completions.create(model="vllm_model", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

Works when Triton is running with the OpenAI frontend. Downside: no cancellation, no access to custom tensors,
no gRPC.

## 2. Our `OpenAICompatClient` — same interface, any transport

[openai_compat.py](openai_compat.py) — the same code as with `openai`, but `protocol="grpc"` goes directly to
KServe gRPC (also works on Triton **without** the OpenAI frontend), `protocol="openai"` goes to the frontend.

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
r = client.chat.completions.create(model="vllm_model", messages=[...], max_tokens=64)
for chunk in client.chat.completions.create(model="vllm_model", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

## 3. Native `TritonLLMClient` — full control

[generate_sync.py](generate_sync.py), [stream_async.py](stream_async.py), [chat.py](chat.py) —
`SamplingParams` with all backend parameters, `extra_inputs` for custom tensors, `wait_for_ready`,
model metadata/config, server-side stream cancellation on `break`.

```python
from triton_sdk import TritonLLMClient, SamplingParams

with TritonLLMClient("localhost:8001", protocol="grpc") as client:
    client.wait_for_ready("vllm_model")
    r = client.generate("The capital of France is", SamplingParams(max_tokens=8, temperature=0), model="vllm_model")
    print(r.text)
```

## All paths at once

[quickstart.py](quickstart.py) runs methods 1–3 across all three ports and prints the responses and timings —
handy to confirm the server is alive and see which path is fastest for you.

```bash
uv run python examples/quickstart.py
```

## Load for monitoring

[load.py](load.py) sends N parallel streams, so there's something to look at on the Grafana dashboard (`localhost:3000`):

```bash
uv run python examples/load.py --concurrency 8 --requests 100
```
