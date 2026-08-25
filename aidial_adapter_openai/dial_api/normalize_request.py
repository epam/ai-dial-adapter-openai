"""
Removal of DIAL-specific fields from requests which are proxied to
an upstream Chat Completions API nearly as-is.

DIAL extends the Chat Completions request with a handful of fields of its own
(see https://github.com/epam/ai-dial-sdk/). Upstreams with strict request
validation reject unknown fields, so every such field must be either
translated into its native counterpart or dropped before proxying.
"""

from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.validation import ensure_dict

_REQUEST_CUSTOM_FIELDS = "custom_fields"
_MESSAGE_CUSTOM_CONTENT = "messages[*].custom_content"
_MESSAGE_CUSTOM_FIELDS = "messages[*].custom_fields"
_TOOL_CUSTOM_FIELDS = "tools[*].custom_fields"

# The native counterpart of DIAL's `cache_breakpoint`. Note that DIAL marks
# a whole message, whereas the Chat Completions API marks an individual
# content part and supports no mode other than "explicit".
_PROMPT_CACHE_BREAKPOINT = {"mode": "explicit"}


def normalize_dial_request(request: dict) -> dict:
    """
    Returns a copy of the request with the DIAL-specific fields either
    translated into their native Chat Completions counterparts or dropped.
    """
    ignored: set[str] = set()
    request = {**request}

    custom_fields = _pop_object(
        request, "custom_fields", _REQUEST_CUSTOM_FIELDS
    )
    ignored.update(f"{_REQUEST_CUSTOM_FIELDS}.{key}" for key in custom_fields)

    if (messages := request.get("messages")) is not None:
        request["messages"] = [
            _normalize_message(ensure_dict("message", message), ignored)
            for message in messages
        ]

    if (tools := request.get("tools")) is not None:
        request["tools"] = [
            _normalize_tool(ensure_dict("tool", tool), ignored)
            for tool in tools
        ]

    if ignored:
        logger.warning(
            "The following request fields are ignored: "
            + ", ".join(sorted(ignored))
        )

    return request


def _normalize_message(message: dict, ignored: set[str]) -> dict:
    message = {**message}

    custom_content = _pop_object(
        message, "custom_content", _MESSAGE_CUSTOM_CONTENT
    )
    ignored.update(f"{_MESSAGE_CUSTOM_CONTENT}.{key}" for key in custom_content)

    custom_fields = _pop_object(
        message, "custom_fields", _MESSAGE_CUSTOM_FIELDS
    )
    breakpoint = custom_fields.pop("cache_breakpoint", None)
    if breakpoint is not None and not _mark_cache_breakpoint(message):
        ignored.add(f"{_MESSAGE_CUSTOM_FIELDS}.cache_breakpoint")
    ignored.update(f"{_MESSAGE_CUSTOM_FIELDS}.{key}" for key in custom_fields)

    return message


def _normalize_tool(tool: dict, ignored: set[str]) -> dict:
    tool = {**tool}

    # The Chat Completions API supports no cache breakpoints on tools.
    custom_fields = _pop_object(tool, "custom_fields", _TOOL_CUSTOM_FIELDS)
    ignored.update(f"{_TOOL_CUSTOM_FIELDS}.{key}" for key in custom_fields)

    return tool


def _mark_cache_breakpoint(message: dict) -> bool:
    """
    Marks the end of the cacheable prefix at the last content part of the
    message. Returns False if the message has no content part to mark.
    """
    content = message.get("content")
    parts: list[dict]

    if isinstance(content, str) and content:
        parts = [{"type": "text", "text": content}]
    elif isinstance(content, list) and content:
        parts = [*content[:-1], dict(ensure_dict("content part", content[-1]))]
    else:
        return False

    parts[-1]["prompt_cache_breakpoint"] = {**_PROMPT_CACHE_BREAKPOINT}
    message["content"] = parts
    return True


def _pop_object(container: dict, field: str, path: str) -> dict:
    """
    Removes `field` from `container` and returns a shallow copy of its value.
    """
    value = container.pop(field, None)
    return {} if value is None else dict(ensure_dict(path, value))
