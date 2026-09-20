"""Server / model metadata as returned by KServe v2 endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TensorMetadata(BaseModel):
    """Name, datatype and shape of one tensor, as reported by the model metadata endpoint."""

    name: str
    datatype: str
    shape: list[int]


class ModelMetadata(BaseModel):
    """Response of ``GET /v2/models/{model}``: versions, platform, and declared inputs/outputs."""

    model_config = ConfigDict(extra="ignore")

    name: str
    versions: list[str] = Field(default_factory=list)
    platform: str = ""
    inputs: list[TensorMetadata] = Field(default_factory=list)
    outputs: list[TensorMetadata] = Field(default_factory=list)

    @property
    def input_names(self) -> frozenset[str]:
        """Names of all declared input tensors.

        :return: A frozenset of input tensor names.
        """
        return frozenset(tensor.name for tensor in self.inputs)


class ModelConfig(BaseModel):
    """Subset of config.pbtxt needed by the SDK (backend detection, decoupled flag)."""

    model_config = ConfigDict(extra="allow")

    name: str
    platform: str = ""
    backend: str = ""
    max_batch_size: int = 0
    decoupled: bool = False
    input: list[dict[str, Any]] = Field(default_factory=list)
    output: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _lift_decoupled(cls, data: Any) -> Any:
        """Expose Triton's nested transaction-policy flag as a plain top-level bool.

        :param data: Raw payload as received from the server, before field validation.
        :return: The payload with a top-level ``decoupled`` key, derived from
            ``model_transaction_policy.decoupled`` when not already present.
        """
        if isinstance(data, dict) and "decoupled" not in data:
            policy = data.get("model_transaction_policy") or {}
            data = {**data, "decoupled": bool(policy.get("decoupled", False))}
        return data

    @property
    def input_names(self) -> frozenset[str]:
        """Names of all declared input tensors.

        :return: A frozenset of input tensor names.
        """
        return frozenset(tensor["name"] for tensor in self.input)

    @property
    def output_names(self) -> frozenset[str]:
        """Names of all declared output tensors.

        :return: A frozenset of output tensor names.
        """
        return frozenset(tensor["name"] for tensor in self.output)

    def input_datatype(self, name: str) -> str | None:
        """Look up the declared config.pbtxt datatype of one input.

        :param name: Input tensor name.
        :return: The raw ``data_type`` string (e.g. ``"TYPE_INT32"``), or ``None`` if the input is not declared.
        """
        return next((tensor.get("data_type") for tensor in self.input if tensor.get("name") == name), None)


class ServerMetadata(BaseModel):
    """Response of ``GET /v2``: server name, version and supported extensions."""

    model_config = ConfigDict(extra="ignore")

    name: str
    version: str
    extensions: list[str] = Field(default_factory=list)


class RepositoryModel(BaseModel):
    """One entry of the model repository index: name, version and load state."""

    name: str
    version: str | None = None
    state: str | None = None
    reason: str | None = None
