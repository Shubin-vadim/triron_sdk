"""Chat messages -> raw prompt. Triton LLM backends accept plain text, so templating lives client-side."""

from __future__ import annotations

from typing import Protocol

from ..core.optional import require_import
from .models import ChatMessage


class ChatTemplate(Protocol):
    """Anything that can turn a list of chat messages into a raw prompt string."""

    def render(self, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
        """Render a conversation into the raw prompt text sent to the model.

        :param messages: The conversation, in order.
        :param add_generation_prompt: Whether to append the tokens that cue the model to start
            its reply (e.g. an assistant turn marker).
        :return: The rendered prompt string.
        """
        ...

    def count_tokens(self, text: str) -> int | None:
        """Count tokens in a piece of text, for usage reporting.

        :param text: Text to tokenize.
        :return: The token count, or ``None`` when the template has no tokenizer.
        """
        ...


class HFChatTemplate:
    """Uses tokenizer.apply_chat_template from transformers (extra: chat)."""

    def __init__(self, tokenizer_id: str) -> None:
        """Load a Hugging Face tokenizer and its chat template.

        :param tokenizer_id: Hugging Face Hub id or local path of the tokenizer to load.
        :raises MissingDependencyError: If the ``transformers`` package is not installed.
        """
        transformers = require_import("transformers", "chat")
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)

    def render(self, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
        """Render a conversation using the tokenizer's own chat template.

        :param messages: The conversation, in order.
        :param add_generation_prompt: Whether to append the tokens that cue the model to start
            its reply.
        :return: The rendered prompt string.
        """
        payload = [message.model_dump(exclude_none=True) for message in messages]
        return str(
            self._tokenizer.apply_chat_template(payload, tokenize=False, add_generation_prompt=add_generation_prompt)
        )

    def count_tokens(self, text: str) -> int | None:
        """Count tokens using the loaded tokenizer.

        :param text: Text to tokenize.
        :return: The number of tokens ``text`` encodes to.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class SimpleChatTemplate:
    """Dependency-free fallback: role-prefixed lines. Good enough for tests and non-instruct models."""

    def render(self, messages: list[ChatMessage], *, add_generation_prompt: bool = True) -> str:
        """Render a conversation as plain ``role: content`` lines.

        :param messages: The conversation, in order.
        :param add_generation_prompt: Whether to append a trailing ``assistant:`` line.
        :return: The rendered prompt string.
        """
        lines = [f"{message.role.value}: {message.content}" for message in messages]
        if add_generation_prompt:
            lines.append("assistant:")
        return "\n".join(lines)

    def count_tokens(self, text: str) -> int | None:
        """Report that no tokenizer is available.

        :param text: Unused; this template has no tokenizer.
        :return: Always ``None``.
        """
        return None
