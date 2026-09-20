from enum import StrEnum


class Protocol(StrEnum):
    """Wire protocol used to talk to Triton."""

    HTTP = "http"
    GRPC = "grpc"
    OPENAI = "openai"  # Triton OpenAI frontend; only usable through triton_sdk.openai
