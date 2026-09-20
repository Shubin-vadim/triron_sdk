"""Python SDK for NVIDIA Triton Inference Server focused on LLM workloads."""

from importlib.metadata import PackageNotFoundError, version

from triton_sdk.client import AsyncTritonLLMClient, TritonLLMClient
from triton_sdk.core.constants import BackendKind, Protocol
from triton_sdk.core.models import GenerateChunk, GenerateRequest, GenerateResponse, SamplingParams
from triton_sdk.exceptions import TritonError
from triton_sdk.settings import TritonSettings, get_settings


try:
    __version__ = version("triton-sdk")
except PackageNotFoundError:  # pragma: no cover - editable install without metadata
    __version__ = "0.0.0"

__all__ = [
    "AsyncTritonLLMClient",
    "BackendKind",
    "GenerateChunk",
    "GenerateRequest",
    "GenerateResponse",
    "Protocol",
    "SamplingParams",
    "TritonError",
    "TritonLLMClient",
    "TritonSettings",
    "__version__",
    "get_settings",
]
