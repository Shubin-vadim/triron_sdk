# Как обращаться к модели

Все примеры рассчитаны на `docker compose --profile openai up -d` — один контейнер отдаёт три порта:

| Порт | Что | Кому |
|---|---|---|
| **9000** | OpenAI-совместимый API (`/v1/chat/completions`, `/v1/models`) | любой OpenAI-клиент, `curl`, наш `OpenAICompatClient(protocol="openai")` |
| **8000** | KServe v2 HTTP (`/v2/models/<m>/generate`, `/infer`, health) | `curl`, наш `TritonLLMClient(protocol="http")` |
| **8001** | KServe v2 gRPC (`ModelInfer`, `ModelStreamInfer`) | наш `TritonLLMClient(protocol="grpc")` — самый быстрый путь, с отменой |

Модель называется так, как папка в `model_repository/` — здесь `vllm_model`.

## 0. Без кода: curl

```bash
# OpenAI API — шаблон чата применяет сервер
curl -s localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "vllm_model",
  "messages": [{"role": "user", "content": "Назови одну планету."}],
  "max_tokens": 32
}'

# KServe generate — сырой промпт, без шаблона; параметры vLLM — JSON-строкой
curl -s localhost:8000/v2/models/vllm_model/generate -H 'Content-Type: application/json' -d '{
  "text_input": "Столица Франции —",
  "sampling_parameters": "{\"max_tokens\": 8, \"temperature\": 0}",
  "exclude_input_in_output": true
}'

# то же со стримом (SSE)
curl -sN localhost:8000/v2/models/vllm_model/generate_stream -H 'Content-Type: application/json' -d '{
  "text_input": "Считаем: один, два,", "stream": true,
  "sampling_parameters": "{\"max_tokens\": 6, \"temperature\": 0}"
}'

# здоровье и список моделей
curl -s localhost:8000/v2/health/ready -o /dev/null -w '%{http_code}\n'
curl -s localhost:9000/v1/models
```

## 1. Официальный `openai` SDK (ничего нашего не нужно)

[openai_sdk.py](openai_sdk.py) — `uv run --with openai python examples/openai_sdk.py`.

```python
import httpx
from openai import OpenAI

# trust_env=False: иначе HTTP_PROXY может утащить localhost и вернуть 503, а Triton запрос не увидит
client = OpenAI(
    base_url="http://localhost:9000/v1",
    api_key="unused",
    http_client=httpx.Client(trust_env=False),
)
r = client.chat.completions.create(model="vllm_model", messages=[{"role": "user", "content": "Привет!"}])
print(r.choices[0].message.content)
```

Подходит, когда Triton запущен с OpenAI-фронтендом. Минус: нет отмены, нет доступа к кастомным тензорам,
нет gRPC.

## 2. Наш `OpenAICompatClient` — тот же интерфейс, любой транспорт

[openai_compat.py](openai_compat.py) — тот же код, что и с `openai`, но `protocol="grpc"` идёт напрямую в
KServe gRPC (работает и на Triton **без** OpenAI-фронтенда), `protocol="openai"` — во фронтенд.

```python
from triton_sdk.openai import OpenAICompatClient

client = OpenAICompatClient("localhost:8001", protocol="grpc", tokenizer="Qwen/Qwen2.5-0.5B-Instruct")
r = client.chat.completions.create(model="vllm_model", messages=[...], max_tokens=64)
for chunk in client.chat.completions.create(model="vllm_model", messages=[...], stream=True):
    print(chunk.choices[0].delta.content or "", end="")
```

## 3. Нативный `TritonLLMClient` — полный контроль

[generate_sync.py](generate_sync.py), [stream_async.py](stream_async.py), [chat.py](chat.py) —
`SamplingParams` со всеми параметрами бэкенда, `extra_inputs` для кастомных тензоров, `wait_for_ready`,
метаданные/конфиг модели, отмена стрима на сервере при `break`.

```python
from triton_sdk import TritonLLMClient, SamplingParams

with TritonLLMClient("localhost:8001", protocol="grpc") as client:
    client.wait_for_ready("vllm_model")
    r = client.generate("Столица Франции —", SamplingParams(max_tokens=8, temperature=0), model="vllm_model")
    print(r.text)
```

## Все пути разом

[quickstart.py](quickstart.py) прогоняет способы 1–3 по всем трём портам и печатает ответы и тайминги —
удобно, чтобы убедиться, что сервер живой и понять, какой путь быстрее у вас.

```bash
uv run python examples/quickstart.py
```

## Нагрузка для мониторинга

[load.py](load.py) шлёт N параллельных стримов, чтобы на дашборде Grafana (`localhost:3000`) было что смотреть:

```bash
uv run python examples/load.py --concurrency 8 --requests 100
```
