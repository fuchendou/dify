from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from core.model_manager import ModelInstance
from core.model_runtime.entities import (
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageRole,
    TextPromptMessageContent,
)


class BufferMemory(ABC):
    def __init__(self, model_instance: ModelInstance):
        self.model_instance = model_instance

    @abstractmethod
    def get_history_prompt_messages(
        self, max_token_limit: int = 2000, message_limit: int | None = None
    ) -> Sequence[PromptMessage]:
        raise NotImplementedError

    def get_history_prompt_text(
        self,
        human_prefix: str = "Human",
        ai_prefix: str = "Assistant",
        max_token_limit: int = 2000,
        message_limit: int | None = None,
    ) -> str:
        prompt_messages = self.get_history_prompt_messages(max_token_limit=max_token_limit, message_limit=message_limit)

        string_messages = []
        for m in prompt_messages:
            if m.role == PromptMessageRole.USER:
                role = human_prefix
            elif m.role == PromptMessageRole.ASSISTANT:
                role = ai_prefix
            else:
                continue

            if isinstance(m.content, list):
                inner_msg = ""
                for content in m.content:
                    if isinstance(content, TextPromptMessageContent):
                        inner_msg += f"{content.data}\n"
                    elif isinstance(content, ImagePromptMessageContent):
                        inner_msg += "[image]\n"

                string_messages.append(f"{role}: {inner_msg.strip()}")
            else:
                message = f"{role}: {m.content}"
                string_messages.append(message)

        return "\n".join(string_messages)

    def _prune_prompt_messages(
        self,
        model_instance: ModelInstance,
        prompt_messages: list[PromptMessage],
        max_token_limit: int,
        message_limit: int | None,
    ) -> list[PromptMessage]:
        messages = list(prompt_messages)
        if message_limit and message_limit > 0:
            messages = messages[-message_limit:]

        if not messages:
            return []

        current_tokens = model_instance.get_llm_num_tokens(messages)
        if current_tokens <= max_token_limit:
            return messages

        while current_tokens > max_token_limit and len(messages) > 1:
            messages.pop(0)
            current_tokens = model_instance.get_llm_num_tokens(messages)

        return messages
