from typing import List, assert_never

from aidial_sdk.exceptions import RequestValidationError
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)
from openai.types.responses import (
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_input_item_param import (
    ResponseInputItemParam,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)

_NO_FUNCTION_CALLING = "Function calling isn't yet supported."

_NO_REFUSAL = "Refusal messages aren't yet supported."


def _convert_content_part(
    part: ChatCompletionContentPartParam | ContentArrayOfContentPart,
) -> ResponseInputContentParam:
    match part["type"]:
        case "refusal":
            raise RequestValidationError(_NO_REFUSAL)
        case "text":
            return {"type": "input_text", "text": part["text"]}
        case "image_url":
            image_url = part["image_url"]
            return {
                "type": "input_image",
                "image_url": image_url["url"],
                "detail": image_url.get("detail", "auto"),
            }
        case "file":
            raise RequestValidationError("File references aren't supported")
        case "input_audio":
            raise RequestValidationError("Audio messages aren't supported")
        case _:
            assert_never(part["type"])


def _convert_message(
    message: ChatCompletionMessageParam,
) -> ResponseInputItemParam:
    match (role := message["role"]):
        case "user" | "assistant" | "system" | "developer":
            res_content: List[ResponseInputContentParam] = []

            content = message.get("content")
            if content is None:
                raise RequestValidationError(_NO_FUNCTION_CALLING)

            if isinstance(content, str):
                res_content.append({"type": "input_text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    res_content.append(_convert_content_part(item))

            return {"role": role, "content": res_content}
        case "function" | "tool":
            raise RequestValidationError(_NO_FUNCTION_CALLING)
        case _:
            assert_never(role)


def convert_messages(
    messages: List[ChatCompletionMessageParam],
) -> ResponseInputParam:
    return [_convert_message(message) for message in messages]


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


def _convert_output(output: List[ResponseOutputItem]) -> Choice:
    if len(output) != 1:
        raise RequestValidationError(
            "The response output should contain exactly one item."
        )

    text_content = ""

    item = output[0]
    match item:
        case ResponseOutputMessage(content=content):
            for part in content:
                match part:
                    case ResponseOutputText(text=text):
                        text_content += text
                    case ResponseOutputRefusal():
                        pass
                    case _:
                        assert_never(part)
        case (
            ResponseFileSearchToolCall()
            | ResponseFunctionToolCall()
            | ResponseFunctionWebSearch()
            | ResponseComputerToolCall()
            | ResponseReasoningItem()
            | ImageGenerationCall()
            | ResponseCodeInterpreterToolCall()
            | LocalShellCall()
            | McpCall()
            | McpListTools()
            | McpApprovalRequest()
        ):
            raise RequestValidationError(
                f"The response output contains an unsupported item type: {item.type}"
            )
        case _:
            assert_never(item)

    return Choice(
        index=0,
        message={"role": "assistant", "content": text_content},  # type: ignore
        finish_reason="stop",
    )


def convert_response(response: Response) -> ChatCompletion:
    return ChatCompletion(
        id=response.id,
        created=int(response.created_at),
        model=response.model,
        object="chat.completion",
        usage=response.usage and _convert_usage(response.usage),
        choices=[_convert_output(response.output)],
    )
