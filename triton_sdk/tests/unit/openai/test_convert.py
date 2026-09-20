import pytest

from triton_sdk.chat import ChatMessage, Role
from triton_sdk.core.models import FinishReason, GenerateResponse
from triton_sdk.openai import UnsupportedParameterError
from triton_sdk.openai.convert import build_usage, finish_reason, to_chat_messages, to_sampling_params


class WordTemplate:
    def render(self, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
        return " ".join(message.content for message in messages)

    def count_tokens(self, text: str) -> int | None:
        return len(text.split())


def test_to_sampling_params_maps_openai_names() -> None:
    params = to_sampling_params(
        {
            "max_completion_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "stop": "\n",
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "logprobs": True,
            "user": "ignored",
            "stream_options": {"include_usage": True},
            "extra_body": {"top_k": 5, "repetition_penalty": 1.2},
        }
    )
    assert params.max_tokens == 32  # noqa: PLR2004
    assert params.stop == ["\n"]
    assert params.return_log_probs is True
    assert (params.top_k, params.repetition_penalty) == (5, 1.2)
    assert params.seed == 7  # noqa: PLR2004


def test_to_sampling_params_defaults_max_tokens() -> None:
    assert to_sampling_params({}).max_tokens == 256  # noqa: PLR2004


@pytest.mark.parametrize("kwargs", [{"tools": []}, {"response_format": {"type": "json_object"}}, {"n": 2}])
def test_to_sampling_params_rejects_unsupported(kwargs: dict[str, object]) -> None:
    with pytest.raises(UnsupportedParameterError):
        to_sampling_params(kwargs)


def test_to_sampling_params_rejects_unknown_keys() -> None:
    with pytest.raises(TypeError, match="typo"):
        to_sampling_params({"typo": 1})


def test_to_chat_messages_accepts_dicts_parts_and_models() -> None:
    messages = to_chat_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}]},
            ChatMessage(role=Role.ASSISTANT, content="ok"),
        ]
    )
    assert [(message.role, message.content) for message in messages] == [
        (Role.SYSTEM, "be brief"),
        (Role.USER, "hello"),
        (Role.ASSISTANT, "ok"),
    ]


def test_to_chat_messages_rejects_images() -> None:
    with pytest.raises(UnsupportedParameterError, match="image_url"):
        to_chat_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


def test_usage_and_finish_reason_inference() -> None:
    template = WordTemplate()
    usage = build_usage(template, "one two three", "a b")
    assert usage is not None
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (3, 2, 5)

    response = GenerateResponse(model="m", text="a b")
    assert finish_reason(response, usage=usage, max_tokens=2) == "length"  # budget filled -> length
    assert finish_reason(response, usage=usage, max_tokens=10) == "stop"
    assert finish_reason(response, usage=None, max_tokens=1) == "stop"  # no tokenizer -> cannot infer
    explicit = GenerateResponse(model="m", text="a b", finish_reason=FinishReason.LENGTH)
    assert finish_reason(explicit, usage=None, max_tokens=100) == "length"
