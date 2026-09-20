#!/usr/bin/env bash
# Block until Triton is ready to serve. Usage: scripts/wait_for_triton.sh [KSERVE_URL] [TIMEOUT_SEC]
#
# Always waits for KServe HTTP /v2/health/ready (default localhost:8000).
# Also waits for the OpenAI frontend /health/ready (TRITON_OPENAI_URL, default localhost:9000)
# when that port is published — compose --profile openai starts uvicorn a few seconds after
# KServe, and Docker may answer 503 on 9000 until it binds. Native profile does not publish
# 9000; after a short grace the script skips the frontend.
set -euo pipefail

KSERVE_URL="${1:-${TRITON_URL:-http://localhost:8000}}"
TIMEOUT="${2:-600}"
OPENAI_URL="${TRITON_OPENAI_URL:-http://localhost:9000}"
OPENAI_REQUIRED=0
[[ -n "${TRITON_OPENAI_URL:-}" ]] && OPENAI_REQUIRED=1
OPENAI_GRACE_SEC=30

[[ "$KSERVE_URL" == http* ]] || KSERVE_URL="http://${KSERVE_URL}"
[[ "$OPENAI_URL" == http* ]] || OPENAI_URL="http://${OPENAI_URL}"

http_code() {
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 2 --max-time 3 "$1" 2>/dev/null || true)"
  [[ -n "$code" ]] || code=000
  printf '%s' "$code"
}

kserve_ready=0
openai_ready=0
openai_present=0
kserve_ready_at=-1

for ((i = 0; i < TIMEOUT; i += 5)); do
  if [[ "$kserve_ready" -eq 0 ]] && curl -fsS "${KSERVE_URL}/v2/health/ready" >/dev/null 2>&1; then
    kserve_ready=1
    kserve_ready_at=$i
    echo "kserve ready at ${KSERVE_URL}"
  fi

  openai_code="$(http_code "${OPENAI_URL}/health/ready")"
  if [[ "$openai_code" == "200" ]]; then
    openai_ready=1
    openai_present=1
  elif [[ "$openai_code" != "000" ]]; then
    openai_present=1
  fi

  if [[ "$kserve_ready" -eq 1 && "$openai_ready" -eq 1 ]]; then
    echo "openai frontend ready at ${OPENAI_URL}"
    exit 0
  fi

  if [[ "$kserve_ready" -eq 1 && "$openai_ready" -eq 0 && "$openai_present" -eq 0 && "$OPENAI_REQUIRED" -eq 0 ]]; then
    if (( i >= kserve_ready_at + OPENAI_GRACE_SEC )); then
      echo "no openai frontend at ${OPENAI_URL} (skipping)"
      exit 0
    fi
  fi

  sleep 5
done

if [[ "$kserve_ready" -eq 0 ]]; then
  echo "triton not ready at ${KSERVE_URL} after ${TIMEOUT}s" >&2
elif [[ "$openai_ready" -eq 0 ]]; then
  echo "openai frontend not ready at ${OPENAI_URL} after ${TIMEOUT}s" >&2
fi
exit 1
