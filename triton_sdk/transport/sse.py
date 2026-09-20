"""Server-Sent Events parser for /generate_stream and OpenAI-style streams."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


# OpenAI-style streams terminate with a literal `data: [DONE]` event; it ends the stream instead of being parsed.
DONE_MARKER = "[DONE]"


async def parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    """Parse a Server-Sent Events stream into its JSON ``data:`` payloads.

    Ignores comment lines (``:``) and keep-alives, and stops (without yielding) at the literal
    ``data: [DONE]`` event used by OpenAI-style streams.

    :param lines: Async iterator of raw text lines, as produced by an HTTP streaming response.
    :return: An async generator yielding each event's JSON payload as a dict.
    :raises json.JSONDecodeError: If an event's payload is not valid JSON.
    """
    buffer: list[str] = []
    async for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if buffer:
                payload = "\n".join(buffer)
                buffer.clear()
                if payload == DONE_MARKER:
                    return
                yield json.loads(payload)
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            buffer.append(line[5:].lstrip())
    if buffer:
        payload = "\n".join(buffer)
        if payload != DONE_MARKER:
            yield json.loads(payload)
