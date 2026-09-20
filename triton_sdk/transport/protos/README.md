# protos

Сюда генерируются Python-стабы из protobuf Triton:

- grpc_service.proto, model_config.proto из https://github.com/triton-inference-server/common/tree/main/protobuf

Запуск: scripts/gen_protos.sh [TRITON_COMMON_REF] (по умолчанию main).
Папка исключена из ruff/mypy. Сгенерированные .py/.pyi коммитим (пользователям не нужен grpcio-tools);
сами .proto не коммитим — тянем по тегу. Текущие стабы: r25.08.
