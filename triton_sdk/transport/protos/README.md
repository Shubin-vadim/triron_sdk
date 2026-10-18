# protos

Python stubs generated here from Triton's protobuf:

- grpc_service.proto, model_config.proto from https://github.com/triton-inference-server/common/tree/main/protobuf

Run with: scripts/gen_protos.sh [TRITON_COMMON_REF] (defaults to main).
This folder is excluded from ruff/mypy. The generated .py/.pyi files are committed (so users don't need
grpcio-tools); the .proto files themselves are not committed — they're pulled by tag. Current stubs: r25.08.
