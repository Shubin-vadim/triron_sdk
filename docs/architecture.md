# Архитектура

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

## Правила слоёв

- `client` знает про LLM (prompt, chunks, chat) и про то, как выбрать адаптер. Единственное место композиции.
- `backends` знают схему тензоров конкретного бэкенда Triton. Не знают про сеть.
- `transport` знает KServe v2 и ничего про LLM. Все ошибки httpx/grpc заворачиваются в `TritonError`.
- `core` — модели и константы, импортируются всеми.

## Жизненный цикл запроса

1. `client.generate()` собирает `GenerateRequest`, подставляет модель по умолчанию.
2. `_resolve_adapter()` берёт `model_config` (кэш) и выбирает адаптер по именам входов, либо использует `TRITON_BACKEND`.
3. Адаптер строит `list[InferInput]` и список нужных выходов.
4. Транспорт делает запрос; для стрима — итерирует ответы до `triton_final_response`.
5. Адаптер превращает `InferResult` в `GenerateChunk` (vLLM: кумулятивный текст → дельта).
6. Клиент склеивает `GenerateResponse` или отдаёт чанки наружу.

## Отмена и таймауты

- Не-стриминговые запросы ретраятся (`tenacity`) по `RETRY_STATUSES` и ошибкам соединения.
- Стрим никогда не ретраится посередине. Клиент оборачивает генератор транспорта в `contextlib.aclosing`,
  поэтому `break` из `async for` сразу закрывает его: HTTP рвёт соединение, gRPC вызывает `call.cancel()`.
- `TIMEOUT` — на unary-запрос целиком. `STREAM_TIMEOUT` — максимальная пауза **между двумя событиями** стрима
  (idle timeout), а не общая длительность: у долгой генерации нет разумного «общего» лимита.

## HTTP: infer vs generate_stream

`/v2/models/{m}/infer` не умеет стримить decoupled-модели по HTTP, поэтому `HttpTransport.stream_infer`
всегда ходит в `/generate_stream` (generate extension, SSE). Отличия двух форматов спрятаны в
`transport/kserve_json.py`:

| | `/infer` | `/generate[_stream]` |
|---|---|---|
| Тело запроса | `{"inputs": [{name, datatype, shape, data}], "outputs": [...]}` | плоский JSON: `{"text_input": "...", "stream": true}` |
| Ответ | `{"outputs": [{name, datatype, shape, data}], "parameters": {...}}` | плоский JSON: `{"text_output": "...", "triton_final_response": true}` |
| Datatype выходов | приходит явно | угадывается по JSON-типу (`DataType.guess`) |
| Выбор выходов | `outputs` | не поддерживается |
| Ошибка посередине | — | событие `{"error": "..."}` → `InferenceError` |

Транспорт гарантирует, что **последний** `InferResult` стрима помечен `triton_final_response=True`,
даже если сервер этот параметр не прислал (lookahead на одно событие).

## gRPC: ModelInfer / ModelStreamInfer

`GrpcTransport` использует сгенерированные стабы из `transport/protos/` (коммитим их в репозиторий, чтобы
пользователям не нужен был `grpcio-tools`; обновление — `scripts/gen_protos.sh <tag>`). Конвертация
тензоров живёт в `transport/grpc_codec.py`:

- Входы уходят как `raw_input_contents` (little-endian, `struct`), это единственный формат, покрывающий FP16/BF16.
- Выходы принимаются и как `raw_output_contents` (так отвечает Triton), и как типизированные `contents`.
- Метаданные-заголовки приводятся к нижнему регистру (`Authorization` → `authorization`) — требование gRPC.
- `stream_infer` пишет один запрос в bidi-стрим и читает ответы до `triton_final_response`; `error_message`
  в ответе → `InferenceError`. Idle-таймаут реализован через `asyncio.wait_for(call.read(), timeout)`, чтобы
  семантика совпадала с HTTP. Если генератор закрыт раньше финала — `call.cancel()`, и Triton прекращает генерацию.

## Адаптеры знают схему модели

`detect_backend(config)` / `get_adapter(kind, config)` создают адаптер **с `ModelConfig`**. Это нужно по трём причинам:

1. **Shape.** Ensemble/BLS TRT-LLM имеют `max_batch_size > 0` → тензоры `[1, N]`; vLLM-бэкенд объявляет
   `max_batch_size: 0` → тензоры `[N]`. Triton отклоняет запрос с неверным рангом.
2. **Фильтрация входов.** Отправляются только те тензоры, которые модель объявляет в `config.input`
   (иначе Triton отвечает `unexpected inference input`). Лишние параметры дропаются с `DEBUG`-логом.
3. **Алиасы.** Имена менялись между релизами бэкенда (`min_length` → `min_tokens`); `resolve_input()` выбирает
   то, что есть в конфиге.

Без конфига (`BackendAdapter()` напрямую) используется каноническая схема бэкенда и `default_batched`.

`request.extra_inputs` — escape hatch для кастомных ансамблей: значения либо готовые `InferInput`, либо скаляр/список.
Тип для голых значений берётся из `config.input[].data_type` (`TYPE_INT32` → `INT32`), затем из
`known_input_types` адаптера (`end_id`, `pad_id`, `lora_task_id`, …), и только потом угадывается по JSON-типу.

## OpenAI-совместимый фасад (`triton_sdk/openai`)

Тонкий слой над `AsyncTritonLLMClient` / `TritonLLMClient`, ничего не знает о транспорте:

```
openai kwargs ──convert.to_sampling_params──▶ SamplingParams
messages      ──convert.to_chat_messages────▶ ChatMessage[] ──chat_template.render──▶ prompt
GenerateResponse ──convert.to_chat_completion──▶ ChatCompletion (usage через chat_template.count_tokens)
GenerateChunk[]  ──streaming.ChatStreamAssembler──▶ role-chunk, delta-chunks, finish-chunk[, usage-chunk]
```

- `types.py` повторяет имена и вложенность `openai.types.*` (`choices[0].message.content`, `choices[0].delta.content`),
  но это наши pydantic-модели, зависимости от пакета `openai` нет.
- `usage` есть только при наличии токенайзера (`tokenizer=` → `HFChatTemplate`). `finish_reason="length"` выводится,
  когда `completion_tokens >= max_tokens`, иначе `"stop"` — бэкенды Triton причину остановки не сообщают.
- Неподдерживаемое (`tools`, `response_format`, `logit_bias`, `n > 1`) → `UnsupportedParameterError`; безвредное
  (`user`, `store`, `service_tier`) игнорируется; опечатки → `TypeError`.
- Сборщики стрима (`streaming/`: `base.py` — общее состояние и подсчёт usage/finish_reason, `chat.py`, `completion.py`) чистые и общие для sync/async — стрим-логика написана один раз.
- Два движка (`engines/`: `base.py` с протоколом `Engine`, `native.py`, `frontend.py`): `NativeEngine` — всё выше поверх `AsyncTritonLLMClient`; `FrontendEngine` — тело запроса
  форвардится в `/v1/chat/completions` OpenAI-фронтенда (`frontend.py`, httpx + SSE с `[DONE]`), ответ валидируется
  в те же типы. Выбор — `protocol="openai"` vs `http|grpc`.
- Клиенты лежат в `client/`: `async_client.py` (`AsyncOpenAICompatClient`, выбирает движок), `sync_client.py`
  (`OpenAICompatClient` — тонкая обёртка над async через `core/sync_runner.AsyncRunner`, свой loop в daemon-потоке;
  тот же runner использует `TritonLLMClient`). Оба лишь собирают движок и разводят по ресурсам — сами ресурсы
  вынесены на уровень выше, в `resources/` (сосед `client/`, `engines/`, `streaming/`), по типу, как в самом
  `openai`: `chat.py`, `completions.py`, `models.py`, в каждом sync- и async-вариант рядом.

## TLS (`transport/tls.py`)

Один `TLSConfig` для всех трёх транспортов, собирается из `TRITON_SSL*`:

| Настройка | gRPC | HTTP / OpenAI-фронтенд |
|---|---|---|
| `SSL=true` или `grpcs://` / `https://` в URL | `secure_channel` | схема `https` |
| `SSL_ROOT_CERT` | `root_certificates` | `verify=<path>` |
| `SSL_CLIENT_CERT` / `SSL_CLIENT_KEY` (mTLS) | `certificate_chain` / `private_key` | `cert=(cert, key)` |
| `SSL_SERVER_NAME_OVERRIDE` | `grpc.ssl_target_name_override` | — |
| `VERIFY_SSL=false` | **не действует** (gRPC всегда проверяет сертификат сервера; пишем WARNING) | `verify=False` |

gRPC-цель всегда `host:port` — `grpc_target()` срезает любую схему, которую пользователь скопировал из HTTP-конфига.
Тесты (`test_tls.py`) поднимают in-process gRPC-сервер на самоподписанных сертификатах из `tests/fixtures/certs`
(TLS, mTLS, отказ без клиентского сертификата, `server_name_override` через 127.0.0.1).

## Что ещё не реализовано (TODO)

- Проверка на живом Triton (см. интеграционные тесты).
