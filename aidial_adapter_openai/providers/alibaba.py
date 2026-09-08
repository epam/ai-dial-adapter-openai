"""
Alibaba Cloud Model Studio specifics of the prompt caching:
https://www.alibabacloud.com/help/en/model-studio/context-cache
"""

from typing import Any, cast

from aidial_client.types.chat import ChatCompletionRequest, Message

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.providers.vendor_adapter import VendorAdapter

_CACHE_CONTROL = {"type": "ephemeral"}

_SESSION_CACHE_HEADERS = {"x-dashscope-session-cache": "enable"}


class AlibabaAdapter(VendorAdapter):
    def get_extra_headers(self, deployment_type: D) -> dict[str, str]:
        # The Responses API doesn't support the `cache_control` markers,
        # therefore the caching is requested for the whole session instead.
        if deployment_type == D.RESPONSES_API:
            return _SESSION_CACHE_HEADERS
        return {}

    def transform_chat_completions_request(
        self, request: ChatCompletionRequest
    ) -> None:
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
