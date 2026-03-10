"""
Coarse-grained prompt truncation that works at the level of the full chat
completion request.

The entire remaining message list is tokenised in a single call via a
:class:`Tokenizer` instance on each truncation step.  This is appropriate
for upstreams (for example, vLLM) that expose a dedicated tokenization endpoint
and can account for all modalities (text, images, files) in one shot.
"""

from typing import (
    Callable,
    List,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

_T = TypeVar("_T")


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal interface required by :func:`truncate_prompt`."""

    async def tokenize(self, request: dict) -> int:
        """Return the token count for *request* (including its ``messages`` field)."""
        ...


async def truncate_prompt(
    tokenizer: Tokenizer,
    original_request: dict,
    messages: List[_T],
    get_raw_message: Callable[[_T], dict],
    max_prompt_tokens: int,
) -> Tuple[List[_T], List[int], int]:
    """Truncate *messages* to fit within *max_prompt_tokens*.

    Token counting is performed by *tokenizer* on the **full** remaining
    message list on every step, so the implementation is modality-agnostic.

    - Fast path: if the full set already fits, return immediately.
    - Otherwise remove the oldest droppable (non-system, non-last) messages
      one by one, re-tokenizing after each removal.
    - If a removed message is an ``assistant`` message with ``tool_calls``,
      the following ``tool`` messages and the next ``assistant`` message are
      also removed (cascade).
    - Raises :`TruncatePromptSystemError` when
      system messages alone exceed the budget.
    - Raises :`TruncatePromptSystemAndLastUserError`
      when system messages + the last non-system message exceed the budget.

    Parameters
    ----------
    tokenizer:
        Object implementing :class:`Tokenizer` — i.e. exposes
        ``async def tokenize(self, request: dict) -> int``.
    original_request:
        The base chat completion request dict (without ``messages``).
        Tools, functions, and other top-level fields are preserved.
    messages:
        Ordered list of message holders to truncate.
    get_raw_message:
        Callable that extracts the plain ``dict`` message from a holder.
    max_prompt_tokens:
        Hard upper bound on total token count.
    """

    all_indices: Set[int] = set(range(len(messages)))

    def _collect(indices: Set[int]) -> List[_T]:
        return [messages[i] for i in sorted(indices)]

    def _build_request(indices: Set[int]) -> dict:
        raw_messages: list[dict] = [
            get_raw_message(messages[i]) for i in sorted(indices)
        ]
        return {**original_request, "messages": raw_messages}

    # Fast path
    prompt_tokens = await tokenizer.tokenize(_build_request(all_indices))
    if prompt_tokens <= max_prompt_tokens:
        return _collect(all_indices), [], prompt_tokens

    system_indices: list[int] = []
    non_system_indices: list[int] = []
    for idx, msg in enumerate(messages):
        if get_raw_message(msg).get("role") == "system":
            system_indices.append(idx)
        else:
            non_system_indices.append(idx)

    system_set: Set[int] = set(system_indices)
    kept: Set[int] = set(all_indices)

    def _cascade_remove_tool_replies(start_idx: int) -> None:
        """Remove consecutive tool replies after *start_idx* and the next assistant."""
        i = start_idx + 1
        while i < len(messages):
            if i not in kept:
                i += 1
                continue
            role = get_raw_message(messages[i]).get("role")
            if role == "tool":
                kept.discard(i)
                i += 1
                continue
            if role == "assistant":
                kept.discard(i)
                break
            break

    last_measured_tokens = prompt_tokens
    for idx in non_system_indices[:-1]:
        if idx not in kept:
            continue

        raw = get_raw_message(messages[idx])
        kept.discard(idx)

        if raw.get("role") == "assistant" and raw.get("tool_calls"):
            _cascade_remove_tool_replies(idx)

        last_measured_tokens = await tokenizer.tokenize(_build_request(kept))
        if last_measured_tokens <= max_prompt_tokens:
            discarded = sorted(all_indices - kept)
            return _collect(kept), discarded, last_measured_tokens

    if non_system_indices:
        if system_set:
            system_tokens = await tokenizer.tokenize(_build_request(system_set))
            if system_tokens > max_prompt_tokens:
                raise TruncatePromptSystemError(
                    max_prompt_tokens, system_tokens
                )

        raise TruncatePromptSystemAndLastUserError(
            max_prompt_tokens, last_measured_tokens
        )
    else:
        raise TruncatePromptSystemError(max_prompt_tokens, last_measured_tokens)
