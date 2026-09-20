# Бэкенды Triton для LLM

| | TensorRT-LLM (`ensemble` / `tensorrt_llm_bls`) | vLLM (`vllm_backend`) | Generic |
|---|---|---|---|
| Определение | есть `max_tokens`, нет `sampling_parameters` | есть `sampling_parameters` или `backend: vllm` | есть `text_input` |
| Prompt | `text_input` BYTES [1,1] | `text_input` BYTES [1] | `text_input` |
| Параметры | отдельные тензоры: `max_tokens` INT32, `temperature` FP32, `top_k` INT32, `top_p` FP32, `stop_words` BYTES [1,N], `bad_words`, `end_id`, `pad_id`, `random_seed`, `return_log_probs` BOOL, ... | `sampling_parameters` — JSON-строка (vLLM `SamplingParams`) | `extra_inputs` как есть |
| Стриминг | `stream` BOOL, decoupled; отдаёт **дельты** | `stream` BOOL, decoupled; отдаёт **дельты** (проверено на 25.08; старые версии — кумулятив, `VLLMAdapter.cumulative_text=True`) | обычно нет |
| Финальный ответ | `triton_final_response=true`, пустой `text_output` | gRPC: `triton_final_response=true`; `generate_stream`: маркера **нет**, конец SSE (SDK сам помечает последний результат) | один ответ |
| Выходы | `text_output`, `cum_log_probs`, `output_log_probs`, `sequence_length` | `text_output` | `text_output` |
| Вход в выходе | `exclude_input_in_output` (в BLS) | `exclude_input_in_output` | — |

## Маппинг `SamplingParams` → TensorRT-LLM

| `SamplingParams` | Тензор | Тип | Примечание |
|---|---|---|---|
| `max_tokens` | `max_tokens` | INT32 | обязательный |
| `temperature` | `temperature` | FP32 | |
| `top_p` / `top_k` | `top_p` / `top_k` | FP32 / INT32 | |
| `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `length_penalty` | одноимённые | FP32 | |
| `min_tokens` | `min_tokens` **или** `min_length` | INT32 | алиас по конфигу модели |
| `beam_width` | `beam_width` | INT32 | |
| `seed` | `random_seed` | UINT64 | |
| `stop` | `stop_words` | BYTES `[1, N]` | |
| `bad_words` | `bad_words` | BYTES `[1, N]` | |
| `return_log_probs` | `return_log_probs` | BOOL | + запрашиваются выходы `cum_log_probs`, `output_log_probs` |
| `ignore_eos` | `end_id = -1` | INT32 | у TRT-LLM нет флага; `-1` в executor = «нет стоп-токена». Не отправляется, если `end_id` задан в `extra_inputs` |
| `request.stream` | `stream` | BOOL | |
| `request.exclude_input_in_output` | `exclude_input_in_output` | BOOL | отправляется только если `False` |
| `request.extra_inputs` | как есть | guess | `end_id`, `pad_id`, `embedding_bias_*`, LoRA и т.д. |

Незаданные (`None`) параметры не отправляются — работают дефолты модели. Тензоры, которых нет в `config.input`,
не отправляются.

## Generate endpoints

`POST /v2/models/{m}/generate` принимает плоский JSON: `{"text_input": "...", "max_tokens": 32, "stream": false}` —
каждое поле верхнего уровня становится входным тензором. Это удобно для GenericAdapter и для отладки, но
для полного контроля (batch, shapes, версии) используем `/infer` и gRPC.

## OpenAI-совместимый фронтенд

Если Triton запущен с `--openai-*` фронтендом (`python/openai` в репозитории сервера), проще использовать
клиент `openai`. Этот SDK нужен для нативного пути: decoupled gRPC, отмена, кастомные ансамбли, метаданные.
