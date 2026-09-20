"""Pure conversions between openai-style arguments/objects and the SDK's own models."""

from __future__ import annotations

import logging
from typing import Any

from ..chat import ChatMessage, ChatTemplate, Role
from ..core.models import FinishReason as SdkFinishReason
from ..core.models import GenerateResponse, SamplingParams
from ..exceptions import TritonError
from .types import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    Completion,
    CompletionChoice,
    CompletionUsage,
    FinishReason,
)


logger = logging.getLogger(__name__)


class UnsupportedParameterError(TritonError):
    """An openai argument that cannot be expressed on a Triton LLM backend."""


# Accepted silently: they only affect OpenAI's billing/routing, never the completion itself.
_IGNORED: frozenset[str] = frozenset(
    {"user", "metadata", "store", "service_tier", "parallel_tool_calls", "stream_options", "timeout", "extra_headers"}
)
# Rejected loudly: honouring them would silently change semantics.
_UNSUPPORTED: frozenset[str] = frozenset(
    {"tools", "tool_choice", "functions", "function_call", "response_format", "logit_bias", "audio", "modalities"}
)


def to_sampling_params(kwargs: dict[str, Any]) -> SamplingParams:
    """Translate openai-style keyword arguments into :class:`SamplingParams`.

    Mutates nothing. Arguments in ``_UNSUPPORTED`` (e.g. ``tools``) and ``n > 1`` raise
    :class:`UnsupportedParameterError`; any other unrecognized key raises :class:`TypeError`.

    :param kwargs: Keyword arguments as passed to an openai-style ``create()`` call.
    :return: The equivalent :class:`~triton_sdk.core.models.SamplingParams`.
    :raises UnsupportedParameterError: If ``kwargs`` uses an argument Triton backends cannot honor.
    :raises TypeError: If ``kwargs`` contains an unrecognized key.
    """
    unsupported = _UNSUPPORTED & kwargs.keys()
    if unsupported:
        msg = f"not supported on Triton backends: {sorted(unsupported)}"
        raise UnsupportedParameterError(msg)
    if kwargs.get("n", 1) not in (None, 1):
        msg = "n > 1 is not supported; issue separate requests"
        raise UnsupportedParameterError(msg)

    known = {
        "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stop", "presence_penalty",
        "frequency_penalty", "logprobs", "top_logprobs", "n", "best_of", "echo", "suffix", "extra_body",
    }  # fmt: skip
    unknown = kwargs.keys() - known - _IGNORED
    if unknown:
        msg = f"unknown arguments: {sorted(unknown)}"
        raise TypeError(msg)

    stop = kwargs.get("stop")
    stop_list = [stop] if isinstance(stop, str) else list(stop or [])
    max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens") or SamplingParams().max_tokens
    extra = dict(kwargs.get("extra_body") or {})

    return SamplingParams(
        max_tokens=max_tokens,
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        seed=kwargs.get("seed"),
        presence_penalty=kwargs.get("presence_penalty"),
        frequency_penalty=kwargs.get("frequency_penalty"),
        stop=stop_list,
        return_log_probs=bool(kwargs.get("logprobs")),
        **{key: extra[key] for key in ("top_k", "repetition_penalty", "min_tokens", "ignore_eos") if key in extra},
    )


def to_chat_messages(messages: list[Any]) -> list[ChatMessage]:
    """Convert openai-style chat messages into :class:`ChatMessage` instances.

    Accepts plain dicts (whose ``content`` may be a list of text parts) or already-built
    :class:`ChatMessage` instances, mixed in the same list.

    :param messages: Messages as passed to an openai-style ``create()`` call.
    :return: The equivalent list of :class:`~triton_sdk.chat.ChatMessage`.
    :raises UnsupportedParameterError: If a message contains a non-text content part (e.g. an image).
    """
    result: list[ChatMessage] = []
    for item in messages:
        if isinstance(item, ChatMessage):
            result.append(item)
            continue
        content = item.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") != "text":
                    msg = f"only text content parts are supported, got {part.get('type')!r}"
                    raise UnsupportedParameterError(msg)
                parts.append(part["text"])
            content = "".join(parts)
        result.append(ChatMessage(role=Role(item["role"]), content=content or "", name=item.get("name")))
    return result


def build_usage(template: ChatTemplate, prompt: str, completion: str) -> CompletionUsage | None:
    """Compute token usage for a prompt/completion pair, if the template can tokenize.

    :param template: Chat template used to count tokens.
    :param prompt: The rendered prompt text.
    :param completion: The generated completion text.
    :return: A :class:`~triton_sdk.openai.types.CompletionUsage`, or ``None`` when the template
        has no tokenizer (:meth:`~triton_sdk.chat.ChatTemplate.count_tokens` returns ``None``).
    """
    prompt_tokens = template.count_tokens(prompt)
    completion_tokens = template.count_tokens(completion)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def finish_reason(response: GenerateResponse, *, usage: CompletionUsage | None, max_tokens: int) -> FinishReason:
    """Map a generation result to an openai-style finish reason.

    Backends rarely report an explicit reason, so ``"length"`` is inferred when the completion
    filled its token budget.

    :param response: The generation result to inspect.
    :param usage: Token usage, if known; used to infer ``"length"`` when the backend is silent.
    :param max_tokens: The token budget that was requested.
    :return: ``"length"`` or ``"stop"``, matching openai's ``FinishReason`` values.
    """
    if response.finish_reason is SdkFinishReason.LENGTH:
        return "length"
    if response.finish_reason is SdkFinishReason.STOP:
        return "stop"
    # Backends rarely report a reason; infer "length" when the completion filled the budget.
    if usage is not None and usage.completion_tokens >= max_tokens:
        return "length"
    return "stop"


def to_chat_completion(
    response: GenerateResponse, *, prompt: str, template: ChatTemplate, max_tokens: int
) -> ChatCompletion:
    """Convert a generation result into an openai-style :class:`ChatCompletion`.

    :param response: The generation result to convert.
    :param prompt: The rendered prompt text, used to compute usage.
    :param template: Chat template used to compute usage.
    :param max_tokens: The token budget that was requested, used to infer the finish reason.
    :return: The equivalent :class:`~triton_sdk.openai.types.ChatCompletion`.
    """
    usage = build_usage(template, prompt, response.text)
    return ChatCompletion(
        model=response.model,
        choices=[
            Choice(
                message=ChatCompletionMessage(content=response.text),
                finish_reason=finish_reason(response, usage=usage, max_tokens=max_tokens),
            )
        ],
        usage=usage,
    )


def to_completion(response: GenerateResponse, *, prompt: str, template: ChatTemplate, max_tokens: int) -> Completion:
    """Convert a generation result into an openai-style legacy :class:`Completion`.

    :param response: The generation result to convert.
    :param prompt: The rendered prompt text, used to compute usage.
    :param template: Chat template used to compute usage.
    :param max_tokens: The token budget that was requested, used to infer the finish reason.
    :return: The equivalent :class:`~triton_sdk.openai.types.Completion`.
    """
    usage = build_usage(template, prompt, response.text)
    return Completion(
        model=response.model,
        choices=[
            CompletionChoice(
                text=response.text, finish_reason=finish_reason(response, usage=usage, max_tokens=max_tokens)
            )
        ],
        usage=usage,
    )
