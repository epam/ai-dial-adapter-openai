"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

from typing import Any, List, Set

from tiktoken import Encoding, encoding_for_model

from aidial_adapter_openai.utils.exceptions import HTTPException


class Tokenizer:
    model: str
    encoding: Encoding

    def __init__(self, model: str) -> None:
        self.model = model
        self.encoding = encoding_for_model(model)

    def calculate_tokens(self, string: str) -> int:
        return len(self.encoding.encode(string))

    def calculate_prompt_tokens(self, messages: List[Any]) -> int:
        return 3 + sum(map(self.calculate_tokens_per_message, messages))

    def calculate_tokens_per_message(self, message: Any) -> int:
        if self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        tokens = tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                tokens += self.calculate_tokens(value)
            if key == "name":
                tokens += tokens_per_name

        return tokens


def discard_messages(
    tokenizer: Tokenizer, messages: List[Any], max_prompt_tokens: int
) -> tuple[List[Any], List[int]]:
    n = len(messages)

    if n == 0:
        return messages, []  # will be rejected by the upstream

    prompt_tokens = 3

    system_messages_count = 0
    kept_messages: Set[int] = set()

    # Count system messages first
    for idx, message in enumerate(messages):
        if message["role"] == "system":
            kept_messages.add(idx)
            system_messages_count += 1
            prompt_tokens += tokenizer.calculate_tokens_per_message(message)

    if max_prompt_tokens < prompt_tokens:
        raise HTTPException(
            message=f"The token size of system messages ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})",
            status_code=400,
            type="invalid_request_error",
        )

    # Then non-system messages in the reverse order
    for idx, message in reversed(list(enumerate(messages))):
        if message["role"] != "system":
            prompt_tokens += tokenizer.calculate_tokens_per_message(message)

            if max_prompt_tokens < prompt_tokens:
                break

            kept_messages.add(idx)

    if (
        len(kept_messages) == system_messages_count
        and system_messages_count != n
    ):
        raise HTTPException(
            message=f"The token size of system messages and the last user message ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})",
            status_code=400,
            type="invalid_request_error",
        )

    new_messages = [
        message for idx, message in enumerate(messages) if idx in kept_messages
    ]

    discarded_messages = list(set(range(n)) - kept_messages)

    return new_messages, discarded_messages
