from typing import Literal, assert_never

from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_function_web_search import (
    Action,
    ActionFind,
    ActionOpenPage,
    ActionSearch,
)

ChatCompletionFinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


def _convert_usage(usage: ResponseUsage) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens
        ),
        completion_tokens_details=CompletionTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens
        ),
    )


def get_usage(response: Response) -> CompletionUsage | None:
    if usage := response.usage:
        return _convert_usage(usage)
    return None


def get_finish_reason(response: Response) -> ChatCompletionFinishReason:
    if response.incomplete_details and (
        reason := response.incomplete_details.reason
    ):
        match reason:
            case "max_output_tokens":
                return "length"
            case "content_filter":
                return "content_filter"
            case _:
                assert_never(reason)

    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall):
            return "tool_calls"

    return "stop"


def get_web_search_action_content(action: Action) -> str:
    match action:
        case ActionSearch(query=query):
            return f"Search '{query}'"
        case ActionOpenPage(url=url):
            return f"Open page {url}"
        case ActionFind(pattern=pattern, url=url):
            return f"Find {pattern!r} in {url}"
        case _:
            return f"Unknown action of type {action.type!r}"
