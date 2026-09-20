#!/usr/bin/env bash
# Generate gRPC stubs for triton_sdk/transport/protos from triton-inference-server/common.
# Usage: scripts/gen_protos.sh [REF]   (REF = tag or branch, default main)
set -euo pipefail

REF="${1:-main}"
BASE_URL="https://raw.githubusercontent.com/triton-inference-server/common/${REF}/protobuf"
OUT_DIR="triton_sdk/transport/protos"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

for proto in grpc_service.proto model_config.proto health.proto; do
  curl -fsSL "${BASE_URL}/${proto}" -o "${TMP_DIR}/${proto}"
done

uv run python -m grpc_tools.protoc \
  -I "$TMP_DIR" \
  --python_out="$OUT_DIR" \
  --pyi_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$TMP_DIR"/*.proto

# protoc emits absolute imports (import grpc_service_pb2); make them package-relative.
sed -i.bak -E 's/^import (\w+_pb2) as/from . import \1 as/' "$OUT_DIR"/*_pb2_grpc.py "$OUT_DIR"/*_pb2.py
rm -f "$OUT_DIR"/*.bak
echo "stubs generated in $OUT_DIR from ${REF}"
