"""Shared fixtures. Unit tests never touch the network."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from triton_sdk.core.models import InferInput, InferResult, ModelConfig, ModelMetadata, RepositoryModel, ServerMetadata
from triton_sdk.settings import TritonSettings
from triton_sdk.transport import TransportInterface


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def settings() -> TritonSettings:
    return TritonSettings(URL="localhost:8000", RETRY_COUNT=1, _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def vllm_config() -> ModelConfig:
    return ModelConfig.model_validate(load_fixture("vllm_model_config.json"))


@pytest.fixture
def trtllm_config() -> ModelConfig:
    return ModelConfig.model_validate(load_fixture("trtllm_model_config.json"))


class FakeTransport(TransportInterface):
    """Scripted transport: returns canned InferResults and records calls."""

    def __init__(self, config: ModelConfig, results: list[InferResult] | None = None) -> None:
        self.config = config
        self.results = results or []
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def server_live(self) -> bool:
        return True

    async def server_ready(self) -> bool:
        return True

    async def model_ready(self, model: str, version: str | None = None) -> bool:
        return True

    async def server_metadata(self) -> ServerMetadata:
        return ServerMetadata(name="triton", version="test")

    async def model_metadata(self, model: str, version: str | None = None) -> ModelMetadata:
        return ModelMetadata(name=model)

    async def model_config(self, model: str, version: str | None = None) -> ModelConfig:
        return self.config

    async def repository_index(self) -> list[RepositoryModel]:
        return [RepositoryModel(name=self.config.name, state="READY")]

    async def infer(self, model: str, inputs: list[InferInput], **kwargs: Any) -> InferResult:
        self.calls.append({"method": "infer", "model": model, "inputs": inputs, **kwargs})
        return self.results[0]

    async def stream_infer(
        self, model: str, inputs: list[InferInput], **kwargs: Any
    ) -> AsyncGenerator[InferResult, None]:  # type: ignore[override]
        self.calls.append({"method": "stream_infer", "model": model, "inputs": inputs, **kwargs})
        for result in self.results:
            yield result

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_transport_factory() -> type[FakeTransport]:
    return FakeTransport
