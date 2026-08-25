"""
Normalization of a DIAL chat completion request into a plain Chat Completions
request, which is proxied to an upstream nearly as-is.

DIAL extends the Chat Completions request with a handful of fields of its own
(see https://github.com/epam/ai-dial-sdk/). Upstreams with strict request
validation reject unknown fields, so every such field is either translated
into its native counterpart or dropped before proxying.

The normalization is split into two independent stages:

1. parsing: DIAL request -> Chat Completions request + DIAL features,
2. absorption: Chat Completions request + DIAL features -> Chat Completions request.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, cast

from aidial_client.types.chat import ChatCompletionRequest as DialRequest
from aidial_client.types.chat import (
    ChatCompletionRequestCustomFields,
    CustomContentParam,
    Message,
    MessageCustomFieldsParam,
    StaticToolParam,
    ToolCustomFieldsParam,
    ToolParam,
)
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    PromptCacheBreakpoint,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as ChatCompletionsRequest,
)

from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.validation import ensure_dict

# The native counterpart of DIAL's `cache_breakpoint`. Note that DIAL marks
# a whole message, whereas the Chat Completions API marks an individual
# content part and supports no mode other than "explicit".
_PROMPT_CACHE_BREAKPOINT: PromptCacheBreakpoint = {"mode": "explicit"}

_T = TypeVar("_T")


@dataclass
class DialFeatures:
    """
    The DIAL-specific fields carried by a DIAL request. The per-message and
    per-tool ones are keyed by the index of the element they came from.
    """

    custom_fields: ChatCompletionRequestCustomFields = field(
        default_factory=ChatCompletionRequestCustomFields
    )
    message_custom_content: dict[int, CustomContentParam] = field(
        default_factory=dict
    )
    message_custom_fields: dict[int, MessageCustomFieldsParam] = field(
        default_factory=dict
    )
    tool_custom_fields: dict[int, ToolCustomFieldsParam] = field(
        default_factory=dict
    )


def _proxy_as_is_on_error(
    normalize: Callable[[dict], dict],
) -> Callable[[dict], dict]:
    """
    Guards against errors in the normalization itself: a request which failed
    to be normalized is proxied to the upstream as-is. The request validation
    errors are reported to the caller as usual.
    """

    @wraps(normalize)
    def wrapper(request: dict) -> dict:
        try:
            return normalize(request)
        except DialException:
            raise
        except Exception:
            logger.exception("Failed to normalize the request")
            return request

    return wrapper


@_proxy_as_is_on_error
def normalize_dial_request(request: dict) -> dict:
    """
    Returns a copy of the request with the DIAL-specific fields either
    translated into their native Chat Completions counterparts or dropped.
    """
    parsed, features = _parse_dial_request(request)
    return cast(dict, _absorb_dial_features(parsed, features))


def _parse_dial_request(
    request: dict,
) -> tuple[ChatCompletionsRequest, DialFeatures]:
    parsed = cast(DialRequest, request).copy()
    features = DialFeatures()

    features.custom_fields = _ensure_dict(
        "custom_fields", parsed.pop("custom_fields", None)
    )

    if (messages := parsed.get("messages")) is not None:
        parsed["messages"] = [
            _parse_message(index, message, features)
            for index, message in enumerate(messages)
        ]

    if (tools := parsed.get("tools")) is not None:
        parsed["tools"] = [
            _parse_tool(index, tool, features)
            for index, tool in enumerate(tools)
        ]

    return cast(ChatCompletionsRequest, parsed), features


def _parse_message(
    index: int, message: Message, features: DialFeatures
) -> Message:
    path = f"messages[{index}]"
    message = _copy_dict(path, message)

    if (custom_content := message.pop("custom_content", None)) is not None:
        features.message_custom_content[index] = _ensure_dict(
            f"{path}.custom_content", custom_content
        )

    if (custom_fields := message.pop("custom_fields", None)) is not None:
        features.message_custom_fields[index] = _ensure_dict(
            f"{path}.custom_fields", custom_fields
        )

    return message


def _parse_tool(
    index: int, tool: ToolParam | StaticToolParam, features: DialFeatures
) -> ToolParam | StaticToolParam:
    path = f"tools[{index}]"
    tool = _copy_dict(path, tool)

    if tool.get("type") == "static_function":
        raise InvalidRequestError(
            f"The upstream doesn't support DIAL static tool at {path!r}"
        )

    # `custom_fields` is only declared on a function tool, but it's taken
    # from a tool of any shape.
    custom_fields = cast(ToolParam, tool).pop("custom_fields", None)
    if custom_fields is not None:
        features.tool_custom_fields[index] = _ensure_dict(
            f"{path}.custom_fields", custom_fields
        )

    return tool


def _absorb_dial_features(
    request: ChatCompletionsRequest, features: DialFeatures
) -> ChatCompletionsRequest:
    # The cache breakpoints are the only DIAL feature with a native
    # counterpart in the Chat Completions API; the rest is ignored.
    breakpoints = {
        index
        for index, custom_fields in features.message_custom_fields.items()
        if custom_fields.get("cache_breakpoint") is not None
    }
    if not breakpoints:
        return request

    request = request.copy()
    request["messages"] = [
        _mark_cache_breakpoint(message) if index in breakpoints else message
        for index, message in enumerate(request.get("messages", []))
    ]

    return request


def _mark_cache_breakpoint(
    message: ChatCompletionMessageParam,
) -> ChatCompletionMessageParam:
    """
    Marks the end of the cacheable prefix at the last content part of the
    message. The message is returned intact, if it has no part to mark.
    """
    content = message.get("content")
    prefix: list

    if isinstance(content, str) and content:
        prefix = []
        part: ChatCompletionContentPartParam = {"type": "text", "text": content}
    elif isinstance(content, list) and content:
        prefix, last = content[:-1], content[-1]
        # A refusal part is the only one which can't carry a breakpoint.
        if last.get("type") == "refusal":
            return message
        part = cast(ChatCompletionContentPartParam, dict(last))
    else:
        return message

    part["prompt_cache_breakpoint"] = _PROMPT_CACHE_BREAKPOINT

    marked = dict(message)
    marked["content"] = [*prefix, part]
    return cast(ChatCompletionMessageParam, marked)


def _ensure_dict(name: str, value: _T | None) -> _T:
    """
    Checks that an optional request element is indeed an object.
    A missing element is treated as an empty one.
    """
    return cast(_T, {} if value is None else ensure_dict(name, value))


def _copy_dict(name: str, value: _T) -> _T:
    """
    Same as `_ensure_dict`, but the element is copied,
    so that the original request is left intact.
    """
    return cast(_T, dict(ensure_dict(name, value)))
