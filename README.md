# Triton-sdk

Python SDK для NVIDIA Triton Inference Server, ориентированный на LLM-нагрузки
(TensorRT-LLM и vLLM бэкенды). Один публичный API — `generate` / `stream` / `chat`,
различия бэкендов и транспортов (HTTP / gRPC) спрятаны внутри.

```python
from triton_sdk import AsyncTritonLLMClient, SamplingParams

async with AsyncTritonLLMClient("localhost:8001", protocol="grpc") as client:
    async for chunk in client.stream("llama3", "Привет!", SamplingParams(max_tokens=64)):
        print(chunk.text, end="")
```

## OpenAI-совместимый интерфейс

Код, написанный под `openai`, переносится заменой конструктора:

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
resp = client.chat.completions.create(model="vllm_model", messages=[{"role": "user", "content": "Привет"}])
resp.choices[0].message.content, resp.usage.total_tokens

for chunk in client.chat.completions.create(model="vllm_model", messages=[...], stream=True):
    chunk.choices[0].delta.content
```

Два движка за одним интерфейсом:

- `protocol="http" | "grpc"` — нативный KServe v2, работает с любым Triton; chat-template и `usage` считаются на клиенте.
- `protocol="openai"` — OpenAI-фронтенд Triton (`/v1/chat/completions`, порт 9000); шаблон и `usage` делает сервер,
`native` недоступен.
`usage` считается токенайзером из `tokenizer=` (без него — `None`). `tools`, `response_format`, `n > 1` не поддерживаются
и вызывают `UnsupportedParameterError`; `extra_body={"top_k": ...}` пробрасывает бэкенд-специфичные параметры.



## Документация

- [docs/usage.md](docs/usage.md) — **руководство по использованию**: установка, выбор клиента и протокола, конфигурация, стриминг и отмена, chat-шаблоны, ошибки, TLS, рецепты
- [examples/README.md](examples/README.md) — как обращаться к модели: curl, `openai`, наш SDK
- [docs/architecture.md](docs/architecture.md) — устройство SDK, слои, решения
- [docs/backends.md](docs/backends.md) — схемы vLLM / TensorRT-LLM, таблица маппинга параметров
- [monitoring/README.md](monitoring/README.md) — Prometheus + Grafana



## Структура


| Путь                         | Роль                                                                                               |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| `triton_sdk/client/`         | Публичный API: `TritonLLMClient` (sync) и `AsyncTritonLLMClient`                                   |
| `triton_sdk/transport/`      | Сетевой слой: `TransportInterface` + `HttpTransport` / `GrpcTransport`, SSE-парсер, protobuf-стабы |
| `triton_sdk/backends/`       | Адаптеры схемы входов/выходов: `tensorrt_llm`, `vllm`, `generic` (generate-endpoint)               |
| `triton_sdk/chat/`           | Chat-шаблоны: список сообщений → сырой prompt, подсчёт токенов                                     |
| `triton_sdk/openai/`         | OpenAI-совместимый фасад: `chat.completions`, `completions`, `models` с типами как в `openai`      |
| `triton_sdk/core/models/`    | Pydantic-модели запросов, ответов, метаданных, тензоров                                            |
| `triton_sdk/core/constants/` | Протоколы, пути KServe v2, коды ошибок, retry-статусы                                              |
| `triton_sdk/settings.py`     | `TritonSettings` (`pydantic-settings`, префикс `TRITON_`)                                          |
| `triton_sdk/exceptions.py`   | Иерархия `TritonError`                                                                             |
| `triton_sdk/tests/`          | `unit/` (моки транспорта), `integration/` (живой Triton, маркер `integration`)                     |
| `examples/`                  | Готовые сценарии использования                                                                     |
| `model_repository/`          | Минимальная vLLM-модель для интеграционных тестов                                                  |
| `monitoring/`                | Prometheus + Grafana (дашборд Triton LLM), см. `monitoring/README.md`                              |




## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv sync --all-extras
cp triton_sdk/config/environment/env.sh.template env.sh && source env.sh
./code_quality_check.sh                                # ruff + mypy + pytest (unit)
```

Интеграционные тесты (нужен GPU и Docker с NVIDIA Container Toolkit):

```bash
docker compose --profile openai up -d   # OpenAI-фронтенд + KServe HTTP/gRPC в одном контейнере
scripts/wait_for_triton.sh
pytest -m integration
```

Мониторинг (Prometheus + Grafana с готовым дашбордом Triton/vLLM):

```bash
docker compose --profile openai --profile monitoring up -d   # Grafana: http://localhost:3000
uv run python examples/load.py --concurrency 8 --requests 200
```

Как обращаться к модели (curl / `openai` / наш SDK) — [examples/README.md](examples/README.md),
`uv run python examples/quickstart.py` прогоняет все пути.

gRPC-стабы генерируются из protobuf Triton: `scripts/gen_protos.sh`.

## Ключевые решения

- Async-ядро, sync-клиент — тонкая обёртка.
- gRPC предпочтителен для стриминга (decoupled mode) и отмены генерации на сервере.
- Бэкенд определяется автоматически по `model_config` (имена входов), можно задать явно.
- Библиотека не настраивает root logger; `triton_sdk.config.configure_logging()` — opt-in для примеров и тестов.

