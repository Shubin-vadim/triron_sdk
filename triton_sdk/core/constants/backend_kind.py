from enum import StrEnum


class BackendKind(StrEnum):
    """Identifies which Triton backend adapter handles a model."""

    TENSORRT_LLM = "tensorrt_llm"
    VLLM = "vllm"
    GENERIC = "generic"  # any model exposing text_input/text_output or the /generate endpoint
