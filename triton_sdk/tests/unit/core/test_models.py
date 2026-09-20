import pytest
from pydantic import ValidationError

from triton_sdk.core.models import GenerateChunk, GenerateResponse, SamplingParams


def test_sampling_params_defaults() -> None:
    params = SamplingParams()
    assert params.max_tokens == 256  # noqa: PLR2004
    assert params.stop == []


def test_sampling_params_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SamplingParams(unknown=1)  # type: ignore[call-arg]


def test_response_from_chunks_concatenates_text() -> None:
    chunks = [GenerateChunk(text="Hel"), GenerateChunk(text="lo", is_final=True)]
    assert GenerateResponse.from_chunks("m", chunks).text == "Hello"


def test_model_config_lifts_decoupled_from_transaction_policy() -> None:
    from triton_sdk.core.models import ModelConfig

    config = ModelConfig.model_validate({"name": "m", "model_transaction_policy": {"decoupled": True}})
    assert config.decoupled is True
    assert ModelConfig(name="m").decoupled is False
