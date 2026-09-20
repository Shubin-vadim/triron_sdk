from collections.abc import AsyncIterator

from triton_sdk.transport.sse import parse_sse


async def _lines(*lines: str) -> AsyncIterator[str]:
    for line in lines:
        yield line


async def test_parse_sse_yields_json_per_event() -> None:
    events = [event async for event in parse_sse(_lines(": keep-alive", 'data: {"a": 1}', "", 'data: {"a": 2}', ""))]
    assert events == [{"a": 1}, {"a": 2}]


async def test_parse_sse_flushes_trailing_event_without_blank_line() -> None:
    events = [event async for event in parse_sse(_lines('data: {"done": true}'))]
    assert events == [{"done": True}]


async def test_parse_sse_stops_at_done_marker() -> None:
    events = [
        event async for event in parse_sse(_lines('data: {"a": 1}', "", "data: [DONE]", "", 'data: {"a": 2}', ""))
    ]
    assert events == [{"a": 1}]
