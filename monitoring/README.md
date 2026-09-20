# Мониторинг

```bash
docker compose --profile openai --profile monitoring up -d
uv run python examples/load.py --concurrency 8 --requests 200   # чтобы было что смотреть
```

| Где | Что |
|---|---|
| http://localhost:3000 | Grafana, дашборд **Triton → Triton LLM** открывается сразу (anonymous admin, без логина) |
| http://localhost:9090 | Prometheus: сырые метрики, `Status → Targets` — видно, кого скрейпит |
| http://localhost:9000/metrics | то, что отдаёт Triton (OpenAI-профиль); http://localhost:8002/metrics — для `native` |

## Откуда берутся метрики

- `nv_*` — ядро Triton: счётчики запросов (`success` / `failure`), длительности по стадиям (queue → input → infer →
  output) в **кумулятивных микросекундах**, очередь, GPU/CPU/память. Есть всегда.
- `vllm:*` — движок vLLM: токены/с (`prompt_tokens_total`, `generation_tokens_total`), гистограммы TTFT,
  inter-token и e2e latency. Появляются только при `REPORT_CUSTOM_METRICS=true` в `config.pbtxt` модели
  (уже включено). **Важно:** vllm_backend 25.08 пробрасывает в Triton только счётчики и гистограммы — gauge-метрик
  vLLM (занятость KV-кэша, running/waiting в планировщике) через Triton нет; за очередью следим по
  `nv_inference_pending_request_count`.
- TRT-LLM-бэкенд отдаёт свои `nv_trt_llm_*` (batch manager, KV-cache blocks, iteration latency) — на дашборде
  их нет, добавить по аналогии.

## Как читать дашборд

- **Request latency by stage** — если растёт `queue`, GPU не успевает (увеличить `max_batch_size`, добавить инстанс
  или GPU); если растёт `infer` — это сама модель (длиннее промпты/ответы).
- **TTFT p95** — то, что видит пользователь как «зависание» перед первым словом; растёт вместе с `waiting`.
- **E2E latency p95** vs **TTFT p95** — разница показывает, сколько времени уходит на генерацию против ожидания;
  если растёт TTFT при стабильном ITL — сервер перегружен на prefill (очередь), если ITL — на decode.
- **Batch efficiency** = 1 для vLLM (он батчит внутри себя, Triton видит по одному запросу) — это нормально;
  для TRT-LLM/не-LLM моделей с dynamic batching должно быть > 1.

Дашборд генерируется скриптом (`monitoring/grafana/dashboards/triton-llm.json`), переменная `$model` фильтрует по
модели, refresh 5 с. Prometheus хранит 2 дня — для прода поднимайте свой.
