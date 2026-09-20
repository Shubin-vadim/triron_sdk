"""GrpcTransport against an in-process grpc.aio server with a scripted GRPCInferenceService. No Triton needed."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest


grpc = pytest.importorskip("grpc", reason="grpc extra not installed")

from triton_sdk.core.constants import FINAL_RESPONSE_PARAM  # noqa: E402
from triton_sdk.core.models import DataType, InferInput  # noqa: E402
from triton_sdk.exceptions import InferenceError, ModelNotFoundError, TritonTimeoutError  # noqa: E402
from triton_sdk.transport import grpc_codec  # noqa: E402
from triton_sdk.transport.grpc import GrpcTransport  # noqa: E402
from triton_sdk.transport.protos import grpc_service_pb2 as pb  # noqa: E402
from triton_sdk.transport.protos import grpc_service_pb2_grpc as pb_grpc  # noqa: E402
from triton_sdk.transport.protos import model_config_pb2 as mc  # noqa: E402


def _text_response(model: str, text: str, *, final: bool = False, request_id: str = "") -> pb.ModelInferResponse:
    response = pb.ModelInferResponse(model_name=model, model_version="1", id=request_id)
    response.outputs.add(name="text_output", datatype="BYTES", shape=[1])
    response.raw_output_contents.append(grpc_codec.encode_tensor(DataType.BYTES, [text]))
    if final:
        response.parameters[FINAL_RESPONSE_PARAM].bool_param = True
    return response


class FakeTriton(pb_grpc.GRPCInferenceServiceServicer):
    """Echo-style LLM: streams the prompt back word by word."""

    def __init__(self) -> None:
        self.stream_cancelled = asyncio.Event()
        self.stream_chunks_sent = 0
        self.received_metadata: dict[str, str] = {}
        self.slow_stream = False

    async def ServerReady(self, request: Any, context: Any) -> pb.ServerReadyResponse:  # noqa: N802
        return pb.ServerReadyResponse(ready=True)

    async def ModelReady(self, request: pb.ModelReadyRequest, context: Any) -> pb.ModelReadyResponse:  # noqa: N802
        return pb.ModelReadyResponse(ready=request.name == "echo")

    async def ModelConfig(self, request: pb.ModelConfigRequest, context: Any) -> pb.ModelConfigResponse:  # noqa: N802
        if request.name != "echo":
            await context.abort(grpc.StatusCode.NOT_FOUND, f"Request for unknown model: {request.name}")
        config = mc.ModelConfig(name="echo", backend="vllm")
        config.model_transaction_policy.decoupled = True
        tensor = config.input.add()
        tensor.name = "text_input"
        return pb.ModelConfigResponse(config=config)

    async def ModelInfer(self, request: pb.ModelInferRequest, context: Any) -> pb.ModelInferResponse:  # noqa: N802
        self.received_metadata = dict(context.invocation_metadata())
        prompt = grpc_codec.decode_tensor(DataType.BYTES, request.raw_input_contents[0])[0]
        if prompt == "explode":
            await context.abort(grpc.StatusCode.INTERNAL, "engine failed")
        return _text_response("echo", f"echo: {prompt}", request_id=request.id)

    async def ModelStreamInfer(  # noqa: N802
        self, request_iterator: AsyncIterator[pb.ModelInferRequest], context: Any
    ) -> AsyncIterator[pb.ModelStreamInferResponse]:
        async for request in request_iterator:
            prompt = grpc_codec.decode_tensor(DataType.BYTES, request.raw_input_contents[0])[0]
            if prompt == "explode":
                yield pb.ModelStreamInferResponse(error_message="engine failed mid-stream")
                return
            words = prompt.split()
            try:
                for index, word in enumerate(words):
                    if self.slow_stream:
                        await asyncio.sleep(0.5)
                    self.stream_chunks_sent += 1
                    final = index == len(words) - 1
                    yield pb.ModelStreamInferResponse(infer_response=_text_response("echo", word, final=final))
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                self.stream_cancelled.set()
                raise


@pytest.fixture
async def server() -> AsyncIterator[tuple[FakeTriton, str]]:
    servicer = FakeTriton()
    grpc_server = grpc.aio.server()
    pb_grpc.add_GRPCInferenceServiceServicer_to_server(servicer, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    await grpc_server.start()
    try:
        yield servicer, f"127.0.0.1:{port}"
    finally:
        await grpc_server.stop(grace=None)


@pytest.fixture
async def transport(server: tuple[FakeTriton, str]) -> AsyncIterator[GrpcTransport]:
    _, address = server
    async with GrpcTransport(address, timeout=5.0, metadata={"Authorization": "Bearer t"}) as instance:
        yield instance  # type: ignore[misc]


def _prompt(text: str) -> list[InferInput]:
    return [InferInput.strings("text_input", [text])]


async def test_health_and_config(transport: GrpcTransport) -> None:
    assert await transport.server_ready() is True
    assert await transport.model_ready("echo") is True
    assert await transport.model_ready("other") is False
    config = await transport.model_config("echo")
    assert config.decoupled is True
    assert config.input_names == {"text_input"}


async def test_unknown_model_maps_to_model_not_found(transport: GrpcTransport) -> None:
    with pytest.raises(ModelNotFoundError) as info:
        await transport.model_config("missing")
    assert info.value.model == "missing"


async def test_infer_roundtrip_and_lowercased_metadata(
    server: tuple[FakeTriton, str], transport: GrpcTransport
) -> None:
    servicer, _ = server
    result = await transport.infer("echo", _prompt("hi"), outputs=["text_output"], request_id="r1")
    assert result.output("text_output").first() == "echo: hi"
    assert result.request_id == "r1"
    assert servicer.received_metadata["authorization"] == "Bearer t"


async def test_infer_server_error_maps_to_inference_error(transport: GrpcTransport) -> None:
    with pytest.raises(InferenceError, match="engine failed"):
        await transport.infer("echo", _prompt("explode"))


async def test_stream_infer_yields_until_final(transport: GrpcTransport) -> None:
    results = [result async for result in transport.stream_infer("echo", _prompt("one two three"))]
    assert [result.output("text_output").first() for result in results] == ["one", "two", "three"]
    assert [result.parameters.get(FINAL_RESPONSE_PARAM, False) for result in results] == [False, False, True]


async def test_stream_infer_error_message_raises(transport: GrpcTransport) -> None:
    with pytest.raises(InferenceError, match="mid-stream"):
        async for _ in transport.stream_infer("echo", _prompt("explode")):
            pass


async def test_stream_infer_break_cancels_server_side(server: tuple[FakeTriton, str], transport: GrpcTransport) -> None:
    servicer, _ = server
    servicer.slow_stream = True
    stream = transport.stream_infer("echo", _prompt("a b c d e f g h"))
    async for result in stream:
        if result.output("text_output").first() == "b":
            break
    await stream.aclose()  # what contextlib.aclosing() does in the client
    await asyncio.wait_for(servicer.stream_cancelled.wait(), timeout=2)
    assert servicer.stream_chunks_sent < 8  # noqa: PLR2004


async def test_stream_infer_idle_timeout(server: tuple[FakeTriton, str], transport: GrpcTransport) -> None:
    servicer, _ = server
    servicer.slow_stream = True  # 0.5s between chunks > 0.1s idle timeout
    with pytest.raises(TritonTimeoutError):
        async for _ in transport.stream_infer("echo", _prompt("a b"), timeout=0.1):
            pass
