from triton_sdk.backends.base import BackendAdapter
from triton_sdk.backends.generic import GenericAdapter
from triton_sdk.backends.tensorrt_llm import TensorRTLLMAdapter
from triton_sdk.backends.vllm import VLLMAdapter
from triton_sdk.core.constants import BackendKind
from triton_sdk.core.models import ModelConfig
from triton_sdk.exceptions import BackendNotSupportedError


__all__ = [
    "BackendAdapter",
    "GenericAdapter",
    "TensorRTLLMAdapter",
    "VLLMAdapter",
    "detect_backend",
    "get_adapter",
]

# Order matters: most specific first, generic last.
_REGISTRY: tuple[type[BackendAdapter], ...] = (VLLMAdapter, TensorRTLLMAdapter, GenericAdapter)
_BY_KIND: dict[BackendKind, type[BackendAdapter]] = {adapter.kind: adapter for adapter in _REGISTRY}


def get_adapter(kind: BackendKind, config: ModelConfig | None = None) -> BackendAdapter:
    """Build the adapter for an explicitly chosen backend.

    :param kind: The backend kind to instantiate.
    :param config: Model config, if available; still drives tensor shapes and input filtering.
    :return: The requested :class:`~triton_sdk.backends.base.BackendAdapter`.
    :raises KeyError: If ``kind`` has no registered adapter.
    """
    return _BY_KIND[kind](config)


def detect_backend(config: ModelConfig) -> BackendAdapter:
    """Pick the adapter whose schema matches a model's declared inputs.

    :param config: Model config to match against the registered adapters, most specific first.
    :return: The first matching :class:`~triton_sdk.backends.base.BackendAdapter`, bound to ``config``.
    :raises BackendNotSupportedError: If no registered adapter matches ``config``.
    """
    for adapter in _REGISTRY:
        if adapter.matches(config):
            return adapter(config)
    msg = f"no adapter for model {config.name!r} (inputs: {sorted(config.input_names)})"
    raise BackendNotSupportedError(msg)
