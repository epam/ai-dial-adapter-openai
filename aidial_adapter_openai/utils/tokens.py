"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""
from typing import Any, List

from tiktoken import Encoding, encoding_for_model

from aidial_adapter_openai.utils.exceptions import HTTPException


def calculate_prompt_tokens(
    messages: List[Any], model: str, encoding: Encoding
) -> int:
    prompt_tokens = 3

    for message in messages:
        prompt_tokens += calculate_tokens_per_message(message, encoding, model)

    return prompt_tokens


def calculate_tokens_per_message(
    message: Any,
    encoding: Encoding,
    model: str,
) -> int:
    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    prompt_tokens = tokens_per_message
    for key, value in message.items():
        prompt_tokens += len(encoding.encode(value))
        if key == "name":
            prompt_tokens += tokens_per_name

    return prompt_tokens


def discard_messages(
    messages: List[Any], model: str, max_prompt_tokens: int
) -> tuple[List[Any], int]:
    if len(messages) == 0:
        return messages, 0  # will be rejected by the upstream

    encoding = encoding_for_model(model)

    prompt_tokens = 3

    non_system_messages_count = 0
    for message in messages:
        if message["role"] != "system":
            non_system_messages_count += 1
            continue

        prompt_tokens += calculate_tokens_per_message(message, encoding, model)

    if max_prompt_tokens < prompt_tokens:
        raise HTTPException(
            message=f"The token size of system messages ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})"
        )

    discarded_messages = non_system_messages_count
    for message in reversed(messages):
        if message["role"] == "system":
            continue

        prompt_tokens += calculate_tokens_per_message(message, encoding, model)

        if max_prompt_tokens < prompt_tokens:
            break

        discarded_messages -= 1

    if (
        discarded_messages == non_system_messages_count
        and non_system_messages_count > 0
    ):
        raise HTTPException(
            message=f"The token size of system messages and the last user message ({prompt_tokens}) exceeds prompt token limit ({max_prompt_tokens})",
            status_code=400,
            type="invalid_request_error",
        )

    messages_without_discarded = []

    remaining_discarded_messages = discarded_messages
    for message in messages:
        if message["role"] == "system":
            messages_without_discarded.append(message)
            continue

        if remaining_discarded_messages > 0:
            remaining_discarded_messages -= 1
        else:
            messages_without_discarded.append(message)

    return messages_without_discarded, discarded_messages
