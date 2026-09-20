"""Generate monitoring/grafana/dashboards/triton-llm.json (Grafana 11 schema)."""

import json
import pathlib

OUT = pathlib.Path(__file__).parent / "dashboards" / "triton-llm.json"

DS = {"type": "prometheus", "uid": "prometheus"}
MODEL = 'model=~"$model"'
_id = 0


def panel(title, exprs, *, unit=None, kind="timeseries", x=0, y=0, w=12, h=8, stack=False, desc=None):
    global _id
    _id += 1
    targets = [{"datasource": DS, "expr": e, "legendFormat": lg, "refId": chr(65 + i)} for i, (e, lg) in enumerate(exprs)]
    p = {
        "id": _id,
        "type": kind,
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    if desc:
        p["description"] = desc
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if kind == "timeseries":
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}
        p["fieldConfig"]["defaults"]["custom"] = {"lineWidth": 2, "fillOpacity": 10}
        if stack:
            p["fieldConfig"]["defaults"]["custom"]["stacking"] = {"mode": "normal"}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"}
    if kind == "gauge":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"]}}
        p["fieldConfig"]["defaults"]["min"] = 0
        p["fieldConfig"]["defaults"]["max"] = 100
    return p


panels = [
    # --- row 1: headline stats ---------------------------------------------------------
    panel("Requests / s", [(f'sum(rate(nv_inference_request_success{{{MODEL}}}[1m]))', "ok")], kind="stat", unit="reqps", x=0, y=0, w=4, h=4),
    panel("Failures / s", [(f'sum(rate(nv_inference_request_failure{{{MODEL}}}[1m]))', "fail")], kind="stat", unit="reqps", x=4, y=0, w=4, h=4),
    panel("Output tokens / s", [(f'sum(rate(vllm:generation_tokens_total{{{MODEL}}}[1m]))', "gen")], kind="stat", unit="short", x=8, y=0, w=4, h=4,
          desc="vllm:* metrics need REPORT_CUSTOM_METRICS=true in config.pbtxt"),
    panel("Pending requests", [(f'sum(nv_inference_pending_request_count{{{MODEL}}})', "pending")], kind="stat", x=12, y=0, w=4, h=4),
    panel("GPU utilization", [('avg(nv_gpu_utilization) * 100', "gpu")], kind="gauge", unit="percent", x=16, y=0, w=4, h=4),
    panel("E2E latency p95", [(f'histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{{{MODEL}}}[1m])))', "p95")], kind="stat", unit="s", x=20, y=0, w=4, h=4),
    # --- row 2: latency ---------------------------------------------------------------------
    panel(
        "Request latency (avg, by stage)",
        [
            (f'sum(rate(nv_inference_queue_duration_us{{{MODEL}}}[1m])) / sum(rate(nv_inference_exec_count{{{MODEL}}}[1m]))', "queue"),
            (f'sum(rate(nv_inference_compute_input_duration_us{{{MODEL}}}[1m])) / sum(rate(nv_inference_exec_count{{{MODEL}}}[1m]))', "input"),
            (f'sum(rate(nv_inference_compute_infer_duration_us{{{MODEL}}}[1m])) / sum(rate(nv_inference_exec_count{{{MODEL}}}[1m]))', "infer"),
            (f'sum(rate(nv_inference_compute_output_duration_us{{{MODEL}}}[1m])) / sum(rate(nv_inference_exec_count{{{MODEL}}}[1m]))', "output"),
        ],
        unit="µs", x=0, y=4, w=12, h=8, stack=True,
        desc="Triton counters are cumulative microseconds; avg = rate(duration) / rate(count)",
    ),
    panel(
        "Time to first token (p50 / p95)",
        [
            (f'histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{{MODEL}}}[1m])))', "p50"),
            (f'histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{{MODEL}}}[1m])))', "p95"),
        ],
        unit="s", x=12, y=4, w=12, h=8,
    ),
    # --- row 3: throughput / concurrency -----------------------------------------------
    panel(
        "Tokens / s",
        [
            (f'sum(rate(vllm:prompt_tokens_total{{{MODEL}}}[1m]))', "prompt"),
            (f'sum(rate(vllm:generation_tokens_total{{{MODEL}}}[1m]))', "generation"),
        ],
        unit="short", x=0, y=12, w=12, h=8,
    ),
    panel(
        "Inter-token latency (p50 / p95)",
        [
            (f'histogram_quantile(0.50, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket{{{MODEL}}}[1m])))', "p50"),
            (f'histogram_quantile(0.95, sum by (le) (rate(vllm:time_per_output_token_seconds_bucket{{{MODEL}}}[1m])))', "p95"),
        ],
        unit="s", x=12, y=12, w=12, h=8,
    ),
    panel(
        "E2E request latency (p50 / p95)",
        [
            (f'histogram_quantile(0.50, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{{{MODEL}}}[1m])))', "p50"),
            (f'histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{{{MODEL}}}[1m])))', "p95"),
        ],
        unit="s", x=0, y=20, w=12, h=8,
        desc="vllm_backend forwards only vLLM counters and histograms to Triton /metrics (no KV-cache / scheduler gauges)",
    ),
    panel(
        "GPU memory",
        [
            ('sum(nv_gpu_memory_used_bytes)', "used"),
            ('sum(nv_gpu_memory_total_bytes)', "total"),
        ],
        unit="bytes", x=12, y=20, w=12, h=8,
    ),
    panel(
        "Requests / s by model",
        [
            (f'sum by (model) (rate(nv_inference_request_success{{{MODEL}}}[1m]))', "{{model}} ok"),
            (f'sum by (model) (rate(nv_inference_request_failure{{{MODEL}}}[1m]))', "{{model}} fail"),
        ],
        unit="reqps", x=0, y=28, w=12, h=8,
    ),
    panel(
        "Batch efficiency (requests per execution)",
        [(f'sum(rate(nv_inference_count{{{MODEL}}}[1m])) / sum(rate(nv_inference_exec_count{{{MODEL}}}[1m]))', "batch")],
        x=12, y=28, w=12, h=8,
        desc="How many requests Triton batched per model execution; 1 = no batching",
    ),
]

dashboard = {
    "uid": "triton-llm",
    "title": "Triton LLM",
    "tags": ["triton", "vllm", "llm"],
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "templating": {
        "list": [
            {
                "name": "model",
                "type": "query",
                "datasource": DS,
                "query": "label_values(nv_inference_request_success, model)",
                "refresh": 2,
                "includeAll": True,
                "multi": True,
                "current": {"text": "All", "value": "$__all"},
            }
        ]
    },
    "panels": panels,
}

OUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
print(OUT, len(panels), "panels")
