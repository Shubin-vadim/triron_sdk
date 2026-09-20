import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from triton_sdk.core.constants import FINAL_RESPONSE_PARAM
from triton_sdk.core.models import DataType, InferInput
from triton_sdk.exceptions import InferenceError, ModelNotFoundError, TritonTimeoutError
from triton_sdk.transport import HttpTransport


BASE = "http://localhost:8000"


def _sse(*events: dict[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _inputs(*, stream: bool = False) -> list[InferInput]:
    return [InferInput.strings("text_input", ["hi"]), InferInput.scalar("stream", stream, DataType.BOOL)]


# --- health / metadata -------------------------------------------------------


async def test_loopback_disables_env_proxy() -> None:
    async with HttpTransport("localhost:8000") as loopback:
        assert loopback._client.trust_env is False
    async with HttpTransport("triton.example:8000") as remote:
        assert remote._client.trust_env is True


@respx.mock
async def test_model_ready_true_on_200() -> None:
    respx.get(f"{BASE}/v2/models/m/ready").mock(return_value=httpx.Response(200))
    async with HttpTransport("localhost:8000") as transport:
        assert await transport.model_ready("m") is True


@respx.mock
async def test_model_config_404_maps_to_model_not_found() -> None:
    respx.get(f"{BASE}/v2/models/missing/config").mock(
        return_value=httpx.Response(404, json={"error": "Request for unknown model: missing"})
    )
    async with HttpTransport("localhost:8000") as transport:
        with pytest.raises(ModelNotFoundError) as info:
            await transport.model_config("missing")
    assert info.value.model == "missing"


# --- infer -------------------------------------------------------------------


@respx.mock
async def test_infer_posts_kserve_body_and_parses_outputs() -> None:
    route = respx.post(f"{BASE}/v2/models/m/versions/2/infer").mock(
        return_value=httpx.Response(
            200,
            json={
                "model_name": "m",
                "model_version": "2",
                "id": "r1",
                "outputs": [{"name": "text_output", "datatype": "BYTES", "shape": [1], "data": ["Hello!"]}],
            },
        )
    )
    async with HttpTransport("localhost:8000") as transport:
        result = await transport.infer("m", _inputs(), outputs=["text_output"], version="2", request_id="r1")

    sent = json.loads(route.calls.last.request.content)
    assert sent["id"] == "r1"
    assert sent["outputs"] == [{"name": "text_output"}]
    assert sent["inputs"][0]["name"] == "text_input"
    assert result.output("text_output").first() == "Hello!"
    assert result.model_version == "2"


@respx.mock
async def test_infer_maps_timeout() -> None:
    respx.post(f"{BASE}/v2/models/m/infer").mock(side_effect=httpx.ReadTimeout("slow"))
    async with HttpTransport("localhost:8000") as transport:
        with pytest.raises(TritonTimeoutError):
            await transport.infer("m", _inputs(), timeout=0.01)


@respx.mock
async def test_infer_500_maps_to_inference_error_with_server_message() -> None:
    respx.post(f"{BASE}/v2/models/m/infer").mock(return_value=httpx.Response(500, json={"error": "engine failed"}))
    async with HttpTransport("localhost:8000") as transport:
        with pytest.raises(InferenceError, match="engine failed"):
            await transport.infer("m", _inputs())


# --- stream_infer ------------------------------------------------------------


@respx.mock
async def test_stream_infer_uses_generate_stream_and_marks_last_final() -> None:
    route = respx.post(f"{BASE}/v2/models/m/generate_stream").mock(
        return_value=httpx.Response(
            200,
            content=_sse({"text_output": "Hel"}, {"text_output": "Hello"}, {"text_output": "Hello!"}),
            headers={"content-type": "text/event-stream"},
        )
    )
    async with HttpTransport("localhost:8000") as transport:
        results = [result async for result in transport.stream_infer("m", _inputs(stream=True))]

    sent = json.loads(route.calls.last.request.content)
    assert sent == {"text_input": "hi", "stream": True}  # flat generate body, no KServe envelope
    assert [result.output("text_output").first() for result in results] == ["Hel", "Hello", "Hello!"]
    assert [result.parameters.get(FINAL_RESPONSE_PARAM, False) for result in results] == [False, False, True]


@respx.mock
async def test_stream_infer_keeps_server_final_flag() -> None:
    respx.post(f"{BASE}/v2/models/m/generate_stream").mock(
        return_value=httpx.Response(
            200, content=_sse({"text_output": "a"}, {"text_output": "", FINAL_RESPONSE_PARAM: True})
        )
    )
    async with HttpTransport("localhost:8000") as transport:
        results = [result async for result in transport.stream_infer("m", _inputs(stream=True))]
    assert len(results) == 2  # noqa: PLR2004
    assert results[-1].parameters[FINAL_RESPONSE_PARAM] is True
    assert FINAL_RESPONSE_PARAM not in results[0].outputs


@respx.mock
async def test_stream_infer_raises_on_mid_stream_error_event() -> None:
    respx.post(f"{BASE}/v2/models/m/generate_stream").mock(
        return_value=httpx.Response(200, content=_sse({"text_output": "a"}, {"error": "OOM"}))
    )
    async with HttpTransport("localhost:8000") as transport:
        stream = transport.stream_infer("m", _inputs(stream=True))
        with pytest.raises(InferenceError, match="OOM"):
            async for _ in stream:
                pass


@respx.mock
async def test_stream_infer_http_error_before_stream() -> None:
    respx.post(f"{BASE}/v2/models/m/generate_stream").mock(
        return_value=httpx.Response(400, json={"error": "model m is not ready"})
    )
    async with HttpTransport("localhost:8000") as transport:
        with pytest.raises(InferenceError.__mro__[1]):  # any TritonError subclass
            async for _ in transport.stream_infer("m", _inputs(stream=True)):
                pass


@respx.mock
async def test_stream_infer_break_closes_connection() -> None:
    consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal consumed
        for i in range(100):
            consumed += 1
            yield f"data: {json.dumps({'text_output': str(i)})}\n\n".encode()

    respx.post(f"{BASE}/v2/models/m/generate_stream").mock(return_value=httpx.Response(200, stream=body()))  # type: ignore[arg-type]
    async with HttpTransport("localhost:8000") as transport:
        stream = transport.stream_infer("m", _inputs(stream=True))
        async for result in stream:
            if result.output("text_output").first() == "2":
                break
        await stream.aclose()  # what contextlib.aclosing() does in the client
    assert consumed < 100  # noqa: PLR2004 - the rest of the body was never read
