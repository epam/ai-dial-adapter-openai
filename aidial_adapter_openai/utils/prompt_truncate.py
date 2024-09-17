from typing import Callable, List, Set, Tuple, TypeVar

from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

_T = TypeVar("_T")

DiscardedMessages = List[int]
UsedTokens = int


def truncate_prompt(
    message_holders: List[_T],
    message_tokens_getter: Callable[[_T], int],
    is_system_message_getter: Callable[[_T], bool],
    max_prompt_tokens: int,
    initial_prompt_tokens: int = 3,
) -> Tuple[List[_T], DiscardedMessages, UsedTokens]:

    prompt_tokens = initial_prompt_tokens

    system_messages_count = 0
    kept_messages: Set[int] = set()

    # Count system messages first
    for idx, message_holder in enumerate(message_holders):
        if is_system_message_getter(message_holder):
            kept_messages.add(idx)
            system_messages_count += 1
            prompt_tokens += message_tokens_getter(message_holder)

    if max_prompt_tokens < prompt_tokens:
        raise TruncatePromptSystemError(max_prompt_tokens, prompt_tokens)

    # Then non-system messages in the reverse order
    for idx, message_holder in reversed(list(enumerate(message_holders))):
        if is_system_message_getter(message_holder):
            continue
        message_tokens = message_tokens_getter(message_holder)

        if max_prompt_tokens < prompt_tokens + message_tokens:
            if len(
                kept_messages
            ) == system_messages_count and system_messages_count != len(
                message_holders
            ):
                raise TruncatePromptSystemAndLastUserError(
                    max_prompt_tokens, prompt_tokens + message_tokens
                )
            break

        prompt_tokens += message_tokens
        kept_messages.add(idx)

    new_messages = [
        message
        for idx, message in enumerate(message_holders)
        if idx in kept_messages
    ]

    discarded_messages = list(set(range(len(message_holders))) - kept_messages)

    return new_messages, discarded_messages, prompt_tokens
