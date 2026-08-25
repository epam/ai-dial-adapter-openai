"""
Alibaba Cloud Model Studio specifics of the prompt caching:
https://www.alibabacloud.com/help/en/model-studio/context-cache
"""

from typing import Any, cast

from aidial_client.types.chat import ChatCompletionRequest, Message

from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)

_CACHE_CONTROL = {"type": "ephemeral"}

_SESSION_CACHE_HEADERS = {"x-dashscope-session-cache": "enable"}


def get_extra_headers(vendor: Vendor, deployment_type: D) -> dict[str, str]:
    if vendor == Vendor.ALIBABA and deployment_type == D.RESPONSES_API:
        return _SESSION_CACHE_HEADERS
    return {}


def convert_chat_completions_request(request: ChatCompletionRequest) -> None:
    messages = request.get("messages") or []

    for message in messages:
        if _pop_cache_breakpoint(message):
            _mark_cache_control(message)

    if _pop_cache_breakpoint(request) and messages:
        _mark_cache_control(messages[-1])


def _pop_cache_breakpoint(fields: ChatCompletionRequest | Message) -> bool:
    if (custom_fields := fields.get("custom_fields")) is None:
        return False

    if custom_fields.pop("cache_breakpoint", None) is None:
        return False

    if not custom_fields:
        del fields["custom_fields"]

    return True


def _mark_cache_control(message: Message) -> None:
    content = message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        message["content"] = content  # type: ignore

    if content:
        # The marker isn't a part of the content part typing,
        # since it's an Alibaba-specific extension.
        cast(dict[str, Any], content[-1])["cache_control"] = _CACHE_CONTROL
