from typing import Callable, Coroutine, List, Set, Tuple, TypeVar

from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

_T = TypeVar("_T")


async def truncate_messages(
    messages: List[_T],
    message_tokens: Callable[[_T], Coroutine[None, None, int]],
    is_system_message: Callable[[_T], bool],
    max_prompt_tokens: int,
    initial_prompt_tokens: int,
) -> Tuple[List[_T], List[int], int]:
    prompt_tokens = initial_prompt_tokens

    system_messages_count = 0
    kept_messages: Set[int] = set()

    # Count system messages first
    for idx, message_holder in enumerate(messages):
        if is_system_message(message_holder):
            kept_messages.add(idx)
            system_messages_count += 1
            prompt_tokens += await message_tokens(message_holder)

    if max_prompt_tokens < prompt_tokens:
        raise TruncatePromptSystemError(max_prompt_tokens, prompt_tokens)

    # Then non-system messages in the reverse order
    for idx, message_holder in reversed(list(enumerate(messages))):
        if is_system_message(message_holder):
            continue
        calculated_message_tokens = await message_tokens(message_holder)

        if max_prompt_tokens < prompt_tokens + calculated_message_tokens:
            if len(kept_messages) == system_messages_count:
                raise TruncatePromptSystemAndLastUserError(
                    max_prompt_tokens, prompt_tokens + calculated_message_tokens
                )
            break

        prompt_tokens += calculated_message_tokens
        kept_messages.add(idx)

    new_messages = [
        message for idx, message in enumerate(messages) if idx in kept_messages
    ]

    discarded_messages = list(set(range(len(messages))) - kept_messages)

    return new_messages, discarded_messages, prompt_tokens
