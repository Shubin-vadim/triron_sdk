# triton-sdk — руководство по использованию

SDK для NVIDIA Triton Inference Server, ориентированный на LLM (бэкенды vLLM и TensorRT-LLM). Один публичный
API поверх HTTP, gRPC и OpenAI-фронтенда Triton. Проверен на Triton 25.08 + vllm_backend.

Содержание:

1. [Установка](#1-установка)
2. [Какой клиент выбрать](#2-какой-клиент-выбрать)
3. [Быстрый старт](#3-быстрый-старт)
4. [Конфигурация](#4-конфигурация)
5. [`TritonLLMClient` — нативный клиент](#5-tritonllmclient--нативный-клиент)
6. [`OpenAICompatClient` — интерфейс как у `openai`](#6-openaicompatclient--интерфейс-как-у-openai)
7. [Параметры генерации](#7-параметры-генерации)
8. [Стриминг и отмена](#8-стриминг-и-отмена)
9. [Chat-шаблоны](#9-chat-шаблоны)
10. [Бэкенды: vLLM и TensorRT-LLM](#10-бэкенды-vllm-и-tensorrt-llm)
11. [Ошибки, таймауты, ретраи](#11-ошибки-таймауты-ретраи)
12. [TLS и аутентификация](#12-tls-и-аутентификация)
13. [Sync vs async](#13-sync-vs-async)
14. [Тестирование своего кода](#14-тестирование-своего-кода)
15. [Логирование](#15-логирование)
16. [Рецепты](#16-рецепты)
17. [Известные ограничения](#17-известные-ограничения)

---

## 1. Установка

```bash
pip install triton-sdk              # HTTP-транспорт, минимум зависимостей (httpx, pydantic, tenacity)
pip install "triton-sdk[grpc]"      # + gRPC (grpcio, protobuf) — рекомендуется для продакшена
pip install "triton-sdk[chat]"      # + transformers для HF chat-шаблонов и подсчёта usage
pip install "triton-sdk[all]"
```

Python ≥ 3.12. gRPC-стабы уже в пакете, `grpcio-tools` не нужен.

## 2. Какой клиент выбрать

| Задача | Клиент |
|---|---|
| Переносите код, написанный под `openai` | `triton_sdk.openai.OpenAICompatClient` |
| Нужен полный контроль: параметры бэкенда, кастомные тензоры, метаданные, отмена | `triton_sdk.TritonLLMClient` |
| Асинхронный сервис (FastAPI, aiohttp) | `AsyncTritonLLMClient` / `AsyncOpenAICompatClient` |

И какой протокол:

| `protocol=` | Порт | Когда |
|---|---|---|
| `"grpc"` | 8001 | **По умолчанию для прода**: в 2–3 раза ниже накладные расходы, отмена генерации на сервере |
| `"http"` | 8000 | Нет `grpcio`, прокси/ingress только HTTP, отладка через curl |
| `"openai"` | 9000 | Только `OpenAICompatClient`; Triton запущен с OpenAI-фронтендом; шаблон и `usage` считает сервер |

## 3. Быстрый старт

```python
from triton_sdk import TritonLLMClient, SamplingParams

with TritonLLMClient("localhost:8001", protocol="grpc") as client:
    client.wait_for_ready("vllm_model", timeout=120)  # модель может грузиться минуты

    response = client.generate(
        "Столица Франции —",
        SamplingParams(max_tokens=32, temperature=0.0),
        model="vllm_model",
    )
    print(response.text)

    for chunk in client.stream("Расскажи анекдот", SamplingParams(max_tokens=128), model="vllm_model"):
        print(chunk.text, end="", flush=True)
```

Или в стиле `openai`:

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
r = client.chat.completions.create(
    model="vllm_model",
    messages=[{"role": "user", "content": "Привет!"}],
    max_tokens=64,
)
print(r.choices[0].message.content, r.usage.total_tokens)
```

Имя модели — имя папки в `model_repository` Triton. Если оно всегда одно, задайте `TRITON_DEFAULT_MODEL` и
`model=` можно не передавать.

## 4. Конфигурация

Все настройки — переменные окружения с префиксом `TRITON_` (или `.env`); любой параметр конструктора их
переопределяет. Полный список — `triton_sdk/config/environment/env.sh.template`.

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `TRITON_URL` | `localhost:8000` | `host:port`; схема `grpcs://` / `https://` включает TLS |
| `TRITON_PROTOCOL` | `http` | `http` / `grpc` / `openai` |
| `TRITON_DEFAULT_MODEL` | — | модель, если не передана в вызове |
| `TRITON_BACKEND` | авто | `vllm` / `tensorrt_llm` / `generic` — принудительный выбор адаптера |
| `TRITON_TIMEOUT` | `60` | таймаут unary-запроса, с |
| `TRITON_STREAM_TIMEOUT` | `300` | **пауза между чанками** стрима, с (не общая длительность) |
| `TRITON_CONNECT_TIMEOUT` | `5` | |
| `TRITON_RETRY_COUNT` / `TRITON_RETRY_START_TIMEOUT` | `3` / `1.0` | ретраи не-стриминговых запросов |
| `TRITON_TOKEN` | — | отправляется как `Authorization: Bearer …` |
| `TRITON_EXTRA_HEADERS` | `{}` | JSON-словарь дополнительных заголовков/метаданных |
| `TRITON_CHAT_TEMPLATE_MODEL` | — | HF tokenizer id для chat-шаблона |
| `TRITON_SSL*` | — | см. [TLS](#12-tls-и-аутентификация) |

Программно — через `TritonSettings`:

```python
from triton_sdk import TritonLLMClient, TritonSettings

settings = TritonSettings(URL="triton.internal:8001", PROTOCOL="grpc", DEFAULT_MODEL="ensemble", TIMEOUT=30)
client = TritonLLMClient(settings=settings)
```

## 5. `TritonLLMClient` — нативный клиент

```python
TritonLLMClient(
    url: str | None = None,            # иначе TRITON_URL
    *,
    protocol: "http" | "grpc" | None,  # иначе TRITON_PROTOCOL
    backend: "vllm" | "tensorrt_llm" | "generic" | None,   # иначе автодетект по конфигу модели
    settings: TritonSettings | None,
    chat_template: ChatTemplate | None,  # см. раздел 9
    transport: TransportInterface | None,  # для тестов: подменить сеть
)
```

Методы (у `AsyncTritonLLMClient` те же, но `await`):

| Метод | Что делает |
|---|---|
| `is_ready(model=None)` | `True`, если сервер (или модель) готов |
| `wait_for_ready(model=None, timeout=300, interval=2)` | Поллит readiness; `ModelNotReadyError` по таймауту |
| `model_metadata(model)` / `model_config(model)` | Метаданные и `config.pbtxt` модели (кэшируется) |
| `list_models()` | Индекс репозитория: имя, версия, состояние |
| `generate(prompt, params=None, *, model=None, model_version=None, request_id=None, timeout=None, extra_inputs=None, exclude_input_in_output=True)` → `GenerateResponse` | Один ответ целиком |
| `stream(prompt, params=None, *, model=None, model_version=None, request_id=None, timeout=None, extra_inputs=None, exclude_input_in_output=True)` → итератор `GenerateChunk` | Токены по мере генерации |
| `chat(messages, params=None, **kw)` / `chat_stream(messages, params=None, **kw)` | То же, но вход — список `ChatMessage`, шаблон применяется на клиенте |
| `render_chat(messages)` → `str` | Посмотреть промпт после шаблона |
| `close()` / контекстный менеджер | Закрыть соединения |

Результаты:

```python
response.text  # str
response.finish_reason  # FinishReason.STOP / LENGTH / UNKNOWN (бэкенды Triton причину не сообщают)
response.cum_log_probs  # float | None, если запрошены log-probs (TRT-LLM)
response.raw  # сырой ответ бэкенда для отладки

chunk.text  # дельта текста
chunk.is_final  # последний чанк стрима
chunk.log_probs  # list[float] | None
```

Важно про decoupled-модели: LLM-модели Triton почти всегда decoupled (стриминговые). Для них `generate()`
**автоматически** ходит через стриминговый API с `stream=false` и собирает ответ — Triton отклоняет обычный
unary `infer` для таких моделей. Вам ничего делать не нужно, но в трассировке gRPC вы увидите `ModelStreamInfer`.

## 6. `OpenAICompatClient` — интерфейс как у `openai`

Те же пути атрибутов и формы ответов, что у пакета `openai` (зависимости от него нет):

```python
from triton_sdk.openai import OpenAICompatClient, AsyncOpenAICompatClient

client = OpenAICompatClient(
    "localhost:8001",
    protocol="grpc",  # "http" | "grpc" — нативный путь; "openai" — фронтенд Triton
    tokenizer="Qwen/Qwen2.5-0.5B-Instruct",  # HF-шаблон чата + подсчёт usage (нужен extra [chat])
)

# chat
r = client.chat.completions.create(model="vllm_model", messages=[...], max_tokens=64, temperature=0.2, stop=["\n\n"])
r.id, r.model, r.choices[0].message.content, r.choices[0].finish_reason, r.usage.prompt_tokens

# стрим — та же последовательность чанков, что у OpenAI: role → дельты → finish_reason → (usage)
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

# модели
[m.id for m in client.models.list().data]

# доступ к нативному клиенту, если нужно что-то за пределами openai-интерфейса
client.native.model_config("vllm_model")  # None при protocol="openai"
```

Поддерживаемые аргументы `create`: `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `seed`,
`stop` (str или list), `presence_penalty`, `frequency_penalty`, `logprobs`, `stream`, `stream_options`,
`extra_body` (бэкенд-специфичное: `{"top_k": 5, "repetition_penalty": 1.2, "min_tokens": 4, "ignore_eos": True}`).

Игнорируются без ошибки: `user`, `metadata`, `store`, `service_tier`, `timeout`, `extra_headers`.
Вызывают `UnsupportedParameterError`: `tools`, `tool_choice`, `functions`, `response_format`, `logit_bias`,
`audio`, `n > 1`. Опечатки в именах → `TypeError`.

Различия двух движков:

| | `protocol="http" / "grpc"` | `protocol="openai"` |
|---|---|---|
| Chat-шаблон | на клиенте (`tokenizer=` или `chat_template=`) | на сервере (`--tokenizer` при запуске фронтенда) |
| `usage` | считается токенайзером клиента, `None` без `tokenizer=`; расходится с сервером на ±1 токен из-за спецтокенов | точное от сервера |
| `finish_reason` | `"length"`, если `completion_tokens >= max_tokens`, иначе `"stop"` | от сервера |
| `native` | доступен | `None` |
| Работает без OpenAI-фронтенда | да | нет |

## 7. Параметры генерации

`SamplingParams` — единый набор для всех бэкендов; `None` = не отправлять, действует дефолт модели.

```python
SamplingParams(
    max_tokens=256,  # обязательный бюджет
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
    return_log_probs=False,  # TRT-LLM: cum_log_probs + output_log_probs; vLLM: игнорируется
    ignore_eos=False,  # vLLM: нативно; TRT-LLM: через end_id=-1
)
```

Как поле превращается в тензор конкретного бэкенда — таблица в [backends.md](backends.md). Параметры, которых
модель не объявляет в конфиге, **не отправляются** (DEBUG-лог), а не ломают запрос.

Помимо `params`, `generate()` / `stream()` принимают `model_version=`, `request_id=` (для трассировки),
`extra_inputs=` (кастомные тензоры, см. рецепты) и `exclude_input_in_output=` (по умолчанию `True` — промпт не
возвращается в ответе).

## 8. Стриминг и отмена

```python
async for chunk in client.stream(prompt, params, model="vllm_model"):
    print(chunk.text, end="")
    if enough:
        break  # ← этого достаточно
```

- `break` (или закрытие генератора) **отменяет запрос на сервере**: по gRPC вызывается `call.cancel()`, и Triton
  прекращает генерацию; по HTTP рвётся соединение (vLLM-бэкенд тоже это замечает).
- Последний чанк всегда помечен `is_final=True`, даже если сервер (HTTP `generate_stream`) маркер не прислал.
- `timeout` у `stream()` — максимальная пауза между двумя чанками, не общая длительность.
- Стрим **не ретраится**; `generate()` ретраится по сетевым ошибкам и 5xx/429 (`RETRY_COUNT`).
- В sync-клиенте `for chunk in client.stream(...)` работает так же; выход из цикла закрывает стрим.

## 9. Chat-шаблоны

vLLM- и TRT-LLM-бэкенды Triton принимают **сырой текст** — шаблон чата нужно применить на клиенте (кроме
`protocol="openai"`, где это делает фронтенд).

```python
from triton_sdk.chat import ChatMessage, Role, HFChatTemplate, SimpleChatTemplate

template = HFChatTemplate("Qwen/Qwen2.5-0.5B-Instruct")  # tokenizer.apply_chat_template, extra [chat]
client = TritonLLMClient("localhost:8001", protocol="grpc", chat_template=template)

messages = [
    ChatMessage(role=Role.SYSTEM, content="Отвечай кратко."),
    ChatMessage(role=Role.USER, content="Что такое Triton?"),
]
client.chat(messages, SamplingParams(max_tokens=64), model="vllm_model")
```

- Без `chat_template=` используется `SimpleChatTemplate` (`role: content` построчно) — годится для тестов, но
  instruct-модели будут отвечать хуже и «продолжать» промпт. Для реальных моделей всегда задавайте HF-шаблон.
- Свой шаблон — любой объект с `render(messages, *, add_generation_prompt=True) -> str` и
  `count_tokens(text) -> int | None` (для `usage`).

## 10. Бэкенды: vLLM и TensorRT-LLM

Адаптер выбирается автоматически по `config.pbtxt` модели (имена входов), `TRITON_BACKEND` принудительно.
Адаптер знает схему модели: строит shape под `max_batch_size`, отправляет только объявленные входы, разрешает
переименования между версиями (`min_length` → `min_tokens`).

| | vLLM (`vllm_backend`) | TensorRT-LLM (`ensemble` / `tensorrt_llm_bls`) |
|---|---|---|
| Определение | вход `sampling_parameters` или `backend: vllm` | входы `text_input` + `max_tokens` |
| Параметры | JSON-строка `sampling_parameters` | тензор на параметр |
| Стрим | дельты | дельты |
| Статус в SDK | **проверен на живом Triton 25.08** | покрыт unit-тестами, живой прогон на вашем деплое обязателен |

Подробности и таблица маппинга — [backends.md](backends.md).

## 11. Ошибки, таймауты, ретраи

Наружу выходят только наследники `triton_sdk.TritonError` (`.message`, `.status_code`, `.model`):

| Исключение | Когда |
|---|---|
| `TritonConnectionError` | сервер недоступен, разрыв соединения, TLS-рукопожатие не удалось |
| `TritonTimeoutError` | превышен `TIMEOUT` / пауза в стриме > `STREAM_TIMEOUT` |
| `ModelNotFoundError` | модель/версия неизвестна (404 / `NOT_FOUND`) |
| `ModelNotReadyError` | модель грузится; `wait_for_ready` по таймауту |
| `InferenceError` | бэкенд отверг или не смог выполнить запрос (в т.ч. ошибка посреди стрима) |
| `StreamCancelledError` | стрим отменён (gRPC `CANCELLED`) |
| `BackendNotSupportedError` | конфиг модели не подошёл ни одному адаптеру |
| `MissingDependencyError` | нужен extra: `pip install "triton-sdk[grpc]"` / `[chat]` |
| `openai.UnsupportedParameterError` | openai-аргумент, который нельзя выразить на Triton |

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

Ретраи: `generate()` повторяет запрос при `TritonConnectionError` и статусах 408/429/500/502/503/504 с
экспоненциальной паузой (`RETRY_COUNT`, `RETRY_START_TIMEOUT`). Стримы не ретраятся.

## 12. TLS и аутентификация

```bash
export TRITON_URL=grpcs://triton.internal:8001     # схема включает TLS; или TRITON_SSL=true
export TRITON_SSL_ROOT_CERT=/etc/ssl/corp-ca.pem    # свой CA; пусто = системные корни
export TRITON_SSL_CLIENT_CERT=/etc/ssl/client.pem   # mTLS
export TRITON_SSL_CLIENT_KEY=/etc/ssl/client.key
export TRITON_SSL_SERVER_NAME_OVERRIDE=triton       # если сертификат выписан не на тот host
export TRITON_TOKEN=...                             # Authorization: Bearer
```

Одни и те же настройки применяются к HTTP, gRPC и OpenAI-фронтенду. `TRITON_VERIFY_SSL=false` отключает
проверку сертификата **только для HTTP** — gRPC всегда проверяет сервер (SDK предупредит в логе).

## 13. Sync vs async

- `AsyncTritonLLMClient` / `AsyncOpenAICompatClient` — ядро; используйте в asyncio-приложениях.
- `TritonLLMClient` / `OpenAICompatClient` — обёртки, которые держат свой event loop в фоновом потоке.
  Безопасно вызывать из обычного кода, **не** вызывайте их из корутины (заблокируете loop).
- Один клиент — один пул соединений; создавайте его один раз на процесс, а не на запрос. Клиенты
  потокобезопасны для параллельных вызовов.

```python
# FastAPI
app.state.llm = AsyncTritonLLMClient("triton:8001", protocol="grpc")


@app.post("/complete")
async def complete(req: Req):
    r = await app.state.llm.generate(req.prompt, SamplingParams(max_tokens=req.max_tokens))
    return {"text": r.text}
```

## 14. Тестирование своего кода

Подмените транспорт — сеть не нужна:

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

    # остальные методы интерфейса — по необходимости


client = AsyncTritonLLMClient(transport=FakeTransport())
```

Готовый `FakeTransport` с записью вызовов есть в `triton_sdk/tests/conftest.py` — можно скопировать.

## 15. Логирование

SDK пишет в логгеры `triton_sdk.*` и **не трогает** root logger. Для отладки:

```python
import logging

logging.getLogger("triton_sdk").setLevel(logging.DEBUG)  # выбор адаптера, дропнутые параметры, отмены
```

`triton_sdk.config.configure_logging("DEBUG")` — удобный opt-in для скриптов (цветной вывод, глушит httpx/grpc).

## 16. Рецепты

**Кастомный ансамбль / дополнительные тензоры** — `extra_inputs=`; тип берётся из `config.pbtxt` модели, для
известных входов TRT-LLM (`end_id`, `pad_id`, `embedding_bias_*`, `lora_task_id`, …) — из таблицы адаптера, иначе
угадывается по значению. Готовый `InferInput` уходит как есть:

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

**Несколько моделей на одном сервере** — один клиент, `model=` в каждом вызове; конфиги кэшируются по имени.

**Выбор версии модели** — `client.generate(..., model_version="2")`; без версии Triton берёт последнюю по
`version_policy` модели.

**Прод-профиль** — gRPC + TLS + `DEFAULT_MODEL` + `STREAM_TIMEOUT` под ваши длины ответов + `RETRY_COUNT=3`;
клиент создаётся один раз при старте процесса; `wait_for_ready()` в readiness-probe вашего сервиса.

**Нагрузка / бенчмарк** — `examples/load.py --concurrency N --requests M`, метрики в Grafana
(`monitoring/README.md`).

## 17. Известные ограничения

- `n > 1`, `tools` / function calling, `response_format` (JSON mode), `logit_bias` — не поддерживаются
  бэкендами Triton через этот SDK; `UnsupportedParameterError`.
- `usage` на нативном пути — оценка токенайзером клиента (±1–2 токена), без `tokenizer=` — `None`.
- `finish_reason` от бэкендов не приходит; `"length"` выводится по бюджету токенов.
- Через Triton не доступны gauge-метрики vLLM (KV-cache, очередь планировщика) — только счётчики/гистограммы.
- TensorRT-LLM-адаптер не верифицирован на живом сервере (см. раздел 10).
- Мультимодальные входы (`image`) — только через `extra_inputs`, в openai-фасаде content-parts кроме `text`
  отклоняются.
