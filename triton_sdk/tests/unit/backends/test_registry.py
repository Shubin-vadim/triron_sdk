from triton_sdk.backends import GenericAdapter, TensorRTLLMAdapter, VLLMAdapter, detect_backend
from triton_sdk.core.models import ModelConfig


def test_detects_vllm(vllm_config: ModelConfig) -> None:
    assert isinstance(detect_backend(vllm_config), VLLMAdapter)


def test_detects_trtllm(trtllm_config: ModelConfig) -> None:
    assert isinstance(detect_backend(trtllm_config), TensorRTLLMAdapter)


def test_falls_back_to_generic() -> None:
    config = ModelConfig(name="custom", input=[{"name": "text_input"}])
    assert isinstance(detect_backend(config), GenericAdapter)
