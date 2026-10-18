# Triton backends for LLMs

| | TensorRT-LLM (`ensemble` / `tensorrt_llm_bls`) | vLLM (`vllm_backend`) | Generic |
|---|---|---|---|
| Detection | has `max_tokens`, no `sampling_parameters` | has `sampling_parameters` or `backend: vllm` | has `text_input` |
| Prompt | `text_input` BYTES [1,1] | `text_input` BYTES [1] | `text_input` |
| Parameters | separate tensors: `max_tokens` INT32, `temperature` FP32, `top_k` INT32, `top_p` FP32, `stop_words` BYTES [1,N], `bad_words`, `end_id`, `pad_id`, `random_seed`, `return_log_probs` BOOL, ... | `sampling_parameters` — a JSON string (vLLM `SamplingParams`) | `extra_inputs` as-is |
| Streaming | `stream` BOOL, decoupled; returns **deltas** | `stream` BOOL, decoupled; returns **deltas** (verified on 25.08; older versions — cumulative, `VLLMAdapter.cumulative_text=True`) | usually none |
| Final response | `triton_final_response=true`, empty `text_output` | gRPC: `triton_final_response=true`; `generate_stream`: **no marker**, end of SSE (the SDK marks the last result itself) | a single response |
| Outputs | `text_output`, `cum_log_probs`, `output_log_probs`, `sequence_length` | `text_output` | `text_output` |
| Input echoed in output | `exclude_input_in_output` (in BLS) | `exclude_input_in_output` | — |

## Mapping `SamplingParams` → TensorRT-LLM

| `SamplingParams` | Tensor | Type | Note |
|---|---|---|---|
| `max_tokens` | `max_tokens` | INT32 | required |
| `temperature` | `temperature` | FP32 | |
| `top_p` / `top_k` | `top_p` / `top_k` | FP32 / INT32 | |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `length_penalty` | same name | FP32 | |
| `min_tokens` | `min_tokens` **or** `min_length` | INT32 | alias depending on model config |
| `beam_width` | `beam_width` | INT32 | |
| `seed` | `random_seed` | UINT64 | |
| `stop` | `stop_words` | BYTES `[1, N]` | |
| `bad_words` | `bad_words` | BYTES `[1, N]` | |
| `return_log_probs` | `return_log_probs` | BOOL | + requests the `cum_log_probs`, `output_log_probs` outputs |
| `ignore_eos` | `end_id = -1` | INT32 | TRT-LLM has no flag for this; `-1` in the executor means "no stop token". Not sent if `end_id` is already set via `extra_inputs` |
| `request.stream` | `stream` | BOOL | |
| `request.exclude_input_in_output` | `exclude_input_in_output` | BOOL | sent only if `False` |
| `request.extra_inputs` | as-is | guessed | `end_id`, `pad_id`, `embedding_bias_*`, LoRA, etc. |

Unset (`None`) parameters are not sent — the model's defaults apply. Tensors not present in `config.input`
are not sent.

## Generate endpoints

`POST /v2/models/{m}/generate` accepts flat JSON: `{"text_input": "...", "max_tokens": 32, "stream": false}` —
every top-level field becomes an input tensor. This is convenient for `GenericAdapter` and for debugging, but
for full control (batch, shapes, versions) we use `/infer` and gRPC.

## OpenAI-compatible frontend

If Triton is running with the `--openai-*` frontend (`python/openai` in the server repository), it's simpler to
use the `openai` client. This SDK is for the native path: decoupled gRPC, cancellation, custom ensembles, metadata.
