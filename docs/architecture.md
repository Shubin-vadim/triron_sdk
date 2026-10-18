# Architecture

```
 caller
   │  generate() / stream() / chat()
   ▼
 client/          AsyncTritonLLMClient ── TritonLLMClient (sync facade, own event loop)
   │  GenerateRequest                      ▲ GenerateChunk / GenerateResponse
   ▼                                       │
 backends/        BackendAdapter: build_inputs() ─────────── parse_result()
   │  list[InferInput]                                        ▲ InferResult
   ▼                                                          │
 transport/       TransportInterface: infer() / stream_infer() / model_config() / ...
                  ├─ HttpTransport  (httpx, KServe v2 JSON + /generate_stream SSE)
                  └─ GrpcTransport  (grpc.aio, ModelInfer / ModelStreamInfer, cancel)
```

## Layer rules

- `client` knows about LLM concerns (prompt, chunks, chat) and how to pick an adapter. The only place where composition happens.
- `backends` know the tensor schema of a specific Triton backend. They know nothing about the network.
- `transport` knows KServe v2 and nothing about LLMs. All httpx/grpc errors are wrapped into `TritonError`.
- `core` — models and constants, imported by everyone.

## Request lifecycle

1. `client.generate()` builds a `GenerateRequest`, substituting the default model.
2. `_resolve_adapter()` fetches `model_config` (cached) and picks an adapter by input names, or uses `TRITON_BACKEND`.
3. The adapter builds a `list[InferInput]` and the list of needed outputs.
4. The transport makes the request; for streaming, it iterates responses up to `triton_final_response`.
5. The adapter turns `InferResult` into a `GenerateChunk` (vLLM: cumulative text → delta).
6. The client assembles a `GenerateResponse` or yields chunks outward.

## Cancellation and timeouts

- Non-streaming requests are retried (`tenacity`) based on `RETRY_STATUSES` and connection errors.
- A stream is never retried mid-flight. The client wraps the transport's generator in `contextlib.aclosing`,
  so a `break` out of `async for` closes it immediately: HTTP tears down the connection, gRPC calls `call.cancel()`.
- `TIMEOUT` applies to the whole unary request. `STREAM_TIMEOUT` is the maximum pause **between two stream events**
  (idle timeout), not the total duration: a long generation has no reasonable "total" limit.

## HTTP: infer vs generate_stream

`/v2/models/{m}/infer` cannot stream decoupled models over HTTP, so `HttpTransport.stream_infer`
always goes to `/generate_stream` (the generate extension, SSE). The differences between the two formats are
hidden in `transport/kserve_json.py`:

| | `/infer` | `/generate[_stream]` |
|---|---|---|
| Request body | `{"inputs": [{name, datatype, shape, data}], "outputs": [...]}` | flat JSON: `{"text_input": "...", "stream": true}` |
| Response | `{"outputs": [{name, datatype, shape, data}], "parameters": {...}}` | flat JSON: `{"text_output": "...", "triton_final_response": true}` |
| Output datatype | sent explicitly | guessed from the JSON type (`DataType.guess`) |
| Output selection | `outputs` | not supported |
| Mid-stream error | — | `{"error": "..."}` event → `InferenceError` |

The transport guarantees that the **last** `InferResult` of a stream is marked `triton_final_response=True`,
even if the server didn't send that parameter (one-event lookahead).

## gRPC: ModelInfer / ModelStreamInfer

`GrpcTransport` uses generated stubs from `transport/protos/` (committed to the repository so
users don't need `grpcio-tools`; update via `scripts/gen_protos.sh <tag>`). Tensor conversion
lives in `transport/grpc_codec.py`:

- Inputs go out as `raw_input_contents` (little-endian, via `struct`), the only format that covers FP16/BF16.
- Outputs are accepted both as `raw_output_contents` (what Triton replies with) and as typed `contents`.
- Metadata headers are lowercased (`Authorization` → `authorization`) — a gRPC requirement.
- `stream_infer` writes a single request into a bidi stream and reads responses up to `triton_final_response`;
  an `error_message` in the response → `InferenceError`. The idle timeout is implemented via
  `asyncio.wait_for(call.read(), timeout)`, to match HTTP semantics. If the generator is closed before the
  final response — `call.cancel()`, and Triton stops generating.

## Adapters know the model schema

`detect_backend(config)` / `get_adapter(kind, config)` create an adapter **with a `ModelConfig`**. This is needed for three reasons:

1. **Shape.** Ensemble/BLS TRT-LLM have `max_batch_size > 0` → tensors `[1, N]`; the vLLM backend declares
   `max_batch_size: 0` → tensors `[N]`. Triton rejects a request with the wrong rank.
2. **Input filtering.** Only tensors that the model declares in `config.input` are sent
   (otherwise Triton responds with `unexpected inference input`). Extra parameters are dropped with a `DEBUG` log.
3. **Aliases.** Names have changed between backend releases (`min_length` → `min_tokens`); `resolve_input()` picks
   whichever is present in the config.

Without a config (`BackendAdapter()` directly), the backend's canonical schema and `default_batched` are used.

`request.extra_inputs` is an escape hatch for custom ensembles: values are either ready-made `InferInput`,
or a scalar/list. The type for raw values is taken from `config.input[].data_type` (`TYPE_INT32` → `INT32`),
then from the adapter's `known_input_types` (`end_id`, `pad_id`, `lora_task_id`, …), and only then guessed from the JSON type.

## OpenAI-compatible facade (`triton_sdk/openai`)

A thin layer over `AsyncTritonLLMClient` / `TritonLLMClient` that knows nothing about the transport:

```
openai kwargs ──convert.to_sampling_params──▶ SamplingParams
messages      ──convert.to_chat_messages────▶ ChatMessage[] ──chat_template.render──▶ prompt
GenerateResponse ──convert.to_chat_completion──▶ ChatCompletion (usage via chat_template.count_tokens)
GenerateChunk[]  ──streaming.ChatStreamAssembler──▶ role-chunk, delta-chunks, finish-chunk[, usage-chunk]
```

- `types.py` mirrors the names and nesting of `openai.types.*` (`choices[0].message.content`, `choices[0].delta.content`),
  but these are our own pydantic models — there is no dependency on the `openai` package.
- `usage` is only available when a tokenizer is present (`tokenizer=` → `HFChatTemplate`). `finish_reason="length"`
  is produced when `completion_tokens >= max_tokens`, otherwise `"stop"` — Triton backends don't report the stop reason.
- Unsupported (`tools`, `response_format`, `logit_bias`, `n > 1`) → `UnsupportedParameterError`; harmless ones
  (`user`, `store`, `service_tier`) are ignored; typos → `TypeError`.
- Stream assemblers (`streaming/`: `base.py` — shared state and usage/finish_reason counting, `chat.py`, `completion.py`) are pure and shared between sync/async — the streaming logic is written once.
- Two engines (`engines/`: `base.py` with the `Engine` protocol, `native.py`, `frontend.py`): `NativeEngine` is everything above layered over `AsyncTritonLLMClient`; `FrontendEngine` forwards the request body
  to the OpenAI frontend's `/v1/chat/completions` (`frontend.py`, httpx + SSE with `[DONE]`), and the response is validated
  into the same types. The choice is `protocol="openai"` vs `http|grpc`.
- Clients live in `client/`: `async_client.py` (`AsyncOpenAICompatClient`, picks the engine), `sync_client.py`
  (`OpenAICompatClient` — a thin wrapper over async via `core/sync_runner.AsyncRunner`, running its own loop in a daemon
  thread; the same runner is used by `TritonLLMClient`). Both only assemble the engine and wire up the resources —
  the resources themselves live one level up, in `resources/` (a sibling of `client/`, `engines/`, `streaming/`), organized
  by type just like in `openai` itself: `chat.py`, `completions.py`, `models.py`, each with its sync and async variant side by side.

## TLS (`transport/tls.py`)

A single `TLSConfig` for all three transports, assembled from `TRITON_SSL*`:

| Setting | gRPC | HTTP / OpenAI frontend |
|---|---|---|
| `SSL=true` or `grpcs://` / `https://` in the URL | `secure_channel` | `https` scheme |
| `SSL_ROOT_CERT` | `root_certificates` | `verify=<path>` |
| `SSL_CLIENT_CERT` / `SSL_CLIENT_KEY` (mTLS) | `certificate_chain` / `private_key` | `cert=(cert, key)` |
| `SSL_SERVER_NAME_OVERRIDE` | `grpc.ssl_target_name_override` | — |
| `VERIFY_SSL=false` | **has no effect** (gRPC always verifies the server certificate; a WARNING is logged) | `verify=False` |

The gRPC target is always `host:port` — `grpc_target()` strips any scheme the user may have copied from the HTTP config.
Tests (`test_tls.py`) spin up an in-process gRPC server with self-signed certificates from `tests/fixtures/certs`
(TLS, mTLS, rejection without a client certificate, `server_name_override` via 127.0.0.1).

## Not yet implemented (TODO)

- Verification against a live Triton (see the integration tests).
