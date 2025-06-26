from typing import List, assert_never

from aidial_sdk.exceptions import RequestValidationError
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_message import (
    Annotation,
    AnnotationURLCitation,
)
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionToolParam,
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolChoiceFunctionParam,
    ToolParam,
)
from openai.types.responses.response_create_params import ToolChoice
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
from openai.types.responses.response_output_text import (
    Annotation as ResponsesAnnotation,
)
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation,
    AnnotationFileCitation,
    AnnotationFilePath,
)
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponsesAnnotationURLCitation,
)

from aidial_adapter_openai.responses.response import (
    get_finish_reason,
    get_usage,
)
from aidial_adapter_openai.utils.log_config import logger

_NO_FUNCTION_CALLING = "Function calling isn't yet supported."

_NO_REFUSAL = "Refusal messages aren't yet supported."


def convert_annotation(annotation: ResponsesAnnotation) -> Annotation | None:
    match annotation:
        case ResponsesAnnotationURLCitation():
            return Annotation(
                type="url_citation",
                url_citation=AnnotationURLCitation(
                    start_index=annotation.start_index,
                    end_index=annotation.end_index,
                    url=annotation.url,
                    title=annotation.title,
                ),
            )
        case (
            AnnotationFileCitation()
            | AnnotationContainerFileCitation()
            | AnnotationFilePath()
        ):
            logger.warning(
                f"Unsupported type of an annotation: {annotation.type}"
            )
            return None
        case _:
            assert_never(annotation)


def convert_tool_choice(
    tool_choice: ChatCompletionToolChoiceOptionParam,
) -> ToolChoice:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict):
        return ToolChoiceFunctionParam(
            type="function",
            name=tool_choice["function"]["name"],
        )
    assert_never(tool_choice)


def convert_tools(tools: List[ChatCompletionToolParam]) -> List[ToolParam]:
    def _convert_tool(tool: ChatCompletionToolParam) -> ToolParam:
        function = tool["function"]
        return FunctionToolParam(
            type="function",
            name=function["name"],
            parameters=function.get("parameters"),
            strict=function.get("strict"),
            description=function.get("description"),
        )

    return [_convert_tool(tool) for tool in tools]


def _convert_output_content_part(
    part: ChatCompletionContentPartParam | ContentArrayOfContentPart,
) -> str:
    match part["type"]:
        case "refusal":
            raise RequestValidationError(_NO_REFUSAL)
        case "text":
            return part["text"]
        case "image_url":
            raise RequestValidationError(
                "Images in the assistant messages aren't supported"
            )
        case "file":
            raise RequestValidationError("File references aren't supported")
        case "input_audio":
            raise RequestValidationError("Audio messages aren't supported")
        case _:
            assert_never(part["type"])


def _convert_input_content_part(
    part: ChatCompletionContentPartParam | ContentArrayOfContentPart,
) -> ResponseInputContentParam:
    match part["type"]:
        case "refusal":
            raise RequestValidationError(_NO_REFUSAL)
        case "text":
            return ResponseInputTextParam(type="input_text", text=part["text"])
        case "image_url":
            image_url = part["image_url"]
            return ResponseInputImageParam(
                type="input_image",
                image_url=image_url["url"],
                detail=image_url.get("detail", "auto"),
            )
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
            content = message.get("content")
            if content is None:
                raise RequestValidationError(_NO_FUNCTION_CALLING)

            if isinstance(content, str):
                parts = [
                    ChatCompletionContentPartTextParam(
                        text=content, type="text"
                    )
                ]
            else:
                parts = content

            if role == "assistant":
                res_content = [
                    _convert_output_content_part(item) for item in parts
                ]

                return EasyInputMessageParam(
                    role=role, content="\n\n".join(res_content)
                )
            else:
                res_content = [
                    _convert_input_content_part(item) for item in parts
                ]
                return EasyInputMessageParam(role=role, content=res_content)

        case "function" | "tool":
            raise RequestValidationError(_NO_FUNCTION_CALLING)
        case _:
            assert_never(role)


def convert_messages(
    messages: List[ChatCompletionMessageParam],
) -> ResponseInputParam:
    return [_convert_message(message) for message in messages]


def _convert_output(output: List[ResponseOutputItem]) -> ChatCompletionMessage:
    if len(output) != 1:
        raise RequestValidationError(
            "The response output should contain exactly one item."
        )

    text_content = ""
    annotations: List[Annotation] = []
    tool_calls: List[ChatCompletionMessageToolCall] = []

    item = output[0]
    match item:
        case ResponseOutputMessage(content=content):
            for part in content:
                match part:
                    case ResponseOutputText(text=text):
                        text_content += text
                        for annotation in part.annotations:
                            if res_annotation := convert_annotation(annotation):
                                annotations.append(res_annotation)
                    case ResponseOutputRefusal():
                        pass
                    case _:
                        assert_never(part)

        case ResponseFunctionToolCall(
            arguments=arguments, name=name, call_id=call_id
        ):
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=call_id,
                    type="function",
                    function=Function(arguments=arguments, name=name),
                )
            )

        case (
            ResponseFileSearchToolCall()
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

    return ChatCompletionMessage(
        role="assistant",
        content=text_content,
        annotations=annotations or None,
        tool_calls=tool_calls or None,
    )


def convert_response(response: Response) -> ChatCompletion:
    message = _convert_output(response.output)

    choice = Choice(
        index=0, message=message, finish_reason=get_finish_reason(response)
    )

    return ChatCompletion(
        id=response.id,
        created=int(response.created_at),
        model=response.model,
        object="chat.completion",
        usage=get_usage(response),
        choices=[choice],
    )
