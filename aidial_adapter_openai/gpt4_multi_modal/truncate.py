from typing import List, Set

from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    MessageTransformResult,
    TruncateTransformedMessagesResult,
)


def truncate_transformed_messages(
    transformations: List[MessageTransformResult], max_prompt_tokens: int
) -> TruncateTransformedMessagesResult:
    # TODO: move it to constant
    overall_tokens = 3
    messages_count = len(transformations)

    system_messages_count = 0
    kept_messages: Set[int] = set()

    # Count system messages first
    for idx, transformation in enumerate(transformations):
        if transformation.message["role"] == "system":
            kept_messages.add(idx)
            system_messages_count += 1
            overall_tokens += transformation.total_tokens

    if max_prompt_tokens < overall_tokens:
        raise TruncatePromptSystemError(max_prompt_tokens, overall_tokens)

    # Then non-system messages in the reverse order
    for idx, transformation in reversed(list(enumerate(transformations))):
        if transformation.message["role"] != "system":
            # If this message won't fit
            if overall_tokens + transformation.total_tokens > max_prompt_tokens:
                break

            overall_tokens += transformation.total_tokens
            kept_messages.add(idx)

    if (
        len(kept_messages) == system_messages_count
        and system_messages_count != messages_count
    ):
        raise TruncatePromptSystemAndLastUserError(
            max_prompt_tokens, overall_tokens
        )

    new_messages = [
        transformation.message
        for idx, transformation in enumerate(transformations)
        if idx in kept_messages
    ]

    discarded_messages = list(set(range(messages_count)) - kept_messages)
    return TruncateTransformedMessagesResult(
        messages=new_messages,
        discarded_messages=discarded_messages,
        overall_token=overall_tokens,
    )
