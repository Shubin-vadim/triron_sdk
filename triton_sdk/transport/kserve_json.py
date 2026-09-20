"""Pure functions: SDK tensor models <-> KServe v2 / generate-extension JSON. No I/O, fully unit-testable."""

from __future__ import annotations

from typing import Any

from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import DataType, InferInput, InferOutput, InferResult
from triton_sdk.exceptions import InferenceError


# Keys of a generate response that are not output tensors.
_GENERATE_RESERVED: frozenset[str] = frozenset({"model_name", "model_version", "id", "parameters", "error"})


def build_infer_body(
    inputs: list[InferInput],
    *,
    outputs: list[str] | None = None,
    request_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /v2/models/{model}/infer``.

    :param inputs: Input tensors to include, in KServe v2 JSON shape.
    :param outputs: Names of output tensors to request; omitted entirely when not given.
    :param request_id: Caller-supplied id to include as ``"id"``.
    :param parameters: Extra request-level parameters to include as ``"parameters"``.
    :return: The request body as a dict, ready to be sent as JSON.
    """
    body: dict[str, Any] = {"inputs": [tensor.to_kserve_json() for tensor in inputs]}
    if outputs:
        body["outputs"] = [{"name": name} for name in outputs]
    if request_id:
        body["id"] = request_id
    if parameters:
        body["parameters"] = parameters
    return body


def build_generate_body(
    inputs: list[InferInput],
    *,
    request_id: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON body for ``POST /v2/models/{model}/generate[_stream]``.

    The generate extension is flat: every top-level key becomes an input tensor. Single-element
    tensors are sent as scalars, everything else as a list. Output selection is not supported by
    the extension.

    :param inputs: Input tensors to flatten into the body.
    :param request_id: Caller-supplied id to include as ``"id"``.
    :param parameters: Extra request-level parameters to include as ``"parameters"``.
    :return: The request body as a dict, ready to be sent as JSON.
    """
    body: dict[str, Any] = {}
    for tensor in inputs:
        body[tensor.name] = tensor.data[0] if len(tensor.data) == 1 else tensor.data
    if request_id:
        body["id"] = request_id
    if parameters:
        body["parameters"] = parameters
    return body


def parse_infer_response(payload: dict[str, Any], *, model: str) -> InferResult:
    """Parse the JSON body returned by ``POST /v2/models/{model}/infer``.

    :param payload: Decoded JSON response body.
    :param model: Model name to fall back to when the payload omits ``model_name``.
    :return: The parsed :class:`~triton_sdk.core.models.InferResult`.
    :raises triton_sdk.exceptions.InferenceError: If the payload contains an ``"error"`` key.
    """
    if "error" in payload:
        raise InferenceError(str(payload["error"]), model=model)
    outputs = {
        item["name"]: InferOutput(
            name=item["name"],
            datatype=DataType(item["datatype"]),
            shape=list(item.get("shape", [])),
            data=list(item.get("data", [])),
        )
        for item in payload.get("outputs", [])
    }
    return InferResult(
        model_name=payload.get("model_name", model),
        model_version=payload.get("model_version"),
        request_id=payload.get("id"),
        outputs=outputs,
        parameters=dict(payload.get("parameters") or {}),
    )


def parse_generate_event(event: dict[str, Any], *, model: str) -> InferResult:
    """Parse one ``/generate`` response or one ``/generate_stream`` SSE event.

    Output tensors are flattened into top-level keys, so their datatype is guessed from the JSON
    value via :meth:`~triton_sdk.core.models.DataType.guess`. Response parameters may appear
    nested under ``"parameters"`` or flattened at the top level (e.g. ``triton_final_response``).

    :param event: Decoded JSON payload of one response or SSE event.
    :param model: Model name to fall back to when the payload omits ``model_name``.
    :return: The parsed :class:`~triton_sdk.core.models.InferResult`.
    :raises triton_sdk.exceptions.InferenceError: If the payload contains an ``"error"`` key.
    """
    if "error" in event:
        raise InferenceError(str(event["error"]), model=model)
    parameters: dict[str, Any] = dict(event.get("parameters") or {})
    outputs: dict[str, InferOutput] = {}
    for key, value in event.items():
        if key in _GENERATE_RESERVED:
            continue
        if key == FINAL_RESPONSE_PARAM:
            parameters[key] = value
            continue
        values = list(value) if isinstance(value, list) else [value]
        outputs[key] = InferOutput(name=key, datatype=DataType.guess(values), shape=[len(values)], data=values)
    return InferResult(
        model_name=event.get("model_name", model),
        model_version=event.get("model_version"),
        request_id=event.get("id"),
        outputs=outputs,
        parameters=parameters,
    )


def mark_final(result: InferResult) -> InferResult:
    """Flag a result as the last response of a stream, if it is not already.

    :param result: The result to flag.
    :return: ``result`` unchanged if already flagged, otherwise a copy with
        ``parameters[triton_sdk.core.constants.FINAL_RESPONSE_PARAM]`` set to ``True``.
    """
    if result.parameters.get(FINAL_RESPONSE_PARAM):
        return result
    return result.model_copy(update={"parameters": {**result.parameters, FINAL_RESPONSE_PARAM: True}})
