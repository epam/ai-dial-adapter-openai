from collections.abc import Generator
from typing import assert_never

import pydantic
from aidial_sdk.chat_completion.enums import Status
from aidial_sdk.chat_completion.request import Attachment, CustomContent, Stage
from aidial_sdk.exceptions import RequestValidationError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionMessageToolCallUnionParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion import Choice
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
    ResponseApplyPatchToolCall,
    ResponseApplyPatchToolCallOutput,
    ResponseCodeInterpreterToolCall,
    ResponseCompactionItem,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseCustomToolCallParam,
    ResponseFileSearchToolCall,
    ResponseFunctionCallOutputItemParam,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputFileContentParam,
    ResponseInputFileParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ToolChoiceAllowedParam,
    ToolChoiceCustomParam,
    ToolChoiceFunctionParam,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.response_computer_tool_call_output_item import (
    ResponseComputerToolCallOutputItem,
)
from openai.types.responses.response_create_params import ToolChoice
from openai.types.responses.response_custom_tool_call_output_item import (
    ResponseCustomToolCallOutputItem,
)
from openai.types.responses.response_function_tool_call_output_item import (
    ResponseFunctionToolCallOutputItem,
)
from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from openai.types.responses.response_output_item import (
    AdditionalTools,
    ImageGenerationCall,
    LocalShellCall,
    LocalShellCallOutput,
    McpApprovalRequest,
    McpApprovalResponse,
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
from openai.types.responses.response_tool_search_call import (
    ResponseToolSearchCall,
)
from openai.types.responses.response_tool_search_output_item import (
    ResponseToolSearchOutputItem,
)

from aidial_adapter_openai.responses.response import (
    get_finish_reason,
    get_usage,
    get_web_search_action_content,
)
from aidial_adapter_openai.utils.log_config import logger

_NO_REFUSAL = "Refusal messages aren't yet supported."

_DEPRECATED_FUNCTION_API = "The deployment doesn't support the deprecated API for functions. Please use tools instead."


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


def parse_response_url_citation(
    annotation: dict,
) -> ResponsesAnnotation | None:
    if annotation.get("type") != "url_citation":
        logger.warning(
            "Unsupported type of an annotation in stream: "
            f"{annotation.get('type')}"
        )
        return None

    try:
        return ResponsesAnnotationURLCitation.model_validate(annotation)
    except pydantic.ValidationError:
        logger.warning(
            f"Failed to parse URL citation annotation in stream: {annotation}"
        )
        return None


def convert_tool_choice(
    tool_choice: ChatCompletionToolChoiceOptionParam,
) -> ToolChoice:
    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, dict):
        match tool_choice["type"]:
            case "allowed_tools":
                return ToolChoiceAllowedParam(
                    type="allowed_tools",
                    mode=tool_choice["allowed_tools"]["mode"],
                    tools=tool_choice["allowed_tools"]["tools"],
                )
            case "function":
                return ToolChoiceFunctionParam(
                    type="function",
                    name=tool_choice["function"]["name"],
                )
            case "custom":
                return ToolChoiceCustomParam(
                    type="custom", name=tool_choice["custom"]["name"]
                )
            case _:
                assert_never(tool_choice["type"])

    assert_never(tool_choice)


_InputToolParam = ChatCompletionToolParam | dict


def convert_tools(tools: list[_InputToolParam]) -> list[ToolParam]:
    _allowed_static_function_names = {"web_search"}

    def _convert_tool(tool: _InputToolParam) -> ToolParam:
        match tool["type"]:
            case "static_function":
                static_function = tool.get("static_function")
                if not static_function:
                    raise RequestValidationError(
                        "Required field 'static_function' is empty or not found."
                    )

                static_function_name = static_function.get("name")
                if static_function_name not in _allowed_static_function_names:
                    msg = (
                        f"Provided static function name ('{static_function_name}') is not supported yet. "
                        f"Allowed values: {list(_allowed_static_function_names)}"
                    )
                    raise RequestValidationError(msg)

                return WebSearchToolParam(
                    type="web_search",
                    **static_function.get("configuration", {}),
                )
            case _:
                function = tool["function"]
                return FunctionToolParam(
                    type="function",
                    name=function["name"],
                    parameters=function.get("parameters"),
                    strict=function.get("strict"),
                    description=function.get("description"),
                )

    return [_convert_tool(tool) for tool in tools]


def _convert_fun_call_part(
    part: ChatCompletionContentPartParam | ContentArrayOfContentPart,
) -> ResponseFunctionCallOutputItemParam:
    match part["type"]:
        case "refusal":
            raise RequestValidationError(_NO_REFUSAL)

        case "text":
            return ResponseInputTextContentParam(
                type="input_text", text=part["text"]
            )

        case "image_url":
            image_url = part["image_url"]
            return ResponseInputImageContentParam(
                type="input_image",
                image_url=image_url["url"],
                detail=image_url.get("detail", "auto"),
            )

        case "file":
            file = part["file"]
            return ResponseInputFileContentParam(
                type="input_file",
                file_id=file.get("file_id"),
                file_data=file.get("file_data"),
                filename=file.get("filename"),
            )

        case "input_audio":
            raise RequestValidationError("Audio messages aren't supported")

        case _:
            assert_never(part["type"])


def _convert_content_part(
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
            file = part["file"]
            item = ResponseInputFileParam(
                type="input_file",
                file_id=file.get("file_id"),
            )

            if (file_data := file.get("file_data")) is not None:
                item["file_data"] = file_data

            if (filename := file.get("filename")) is not None:
                item["filename"] = filename

            return item

        case "input_audio":
            raise RequestValidationError("Audio messages aren't supported")

        case _:
            assert_never(part["type"])


def _convert_tool_call(
    tool_call: ChatCompletionMessageToolCallUnionParam,
) -> ResponseFunctionToolCallParam | ResponseCustomToolCallParam:
    match tool_call["type"]:
        case "function":
            function = tool_call["function"]
            return ResponseFunctionToolCallParam(
                type="function_call",
                call_id=tool_call["id"],
                name=function["name"],
                arguments=function["arguments"],
            )
        case "custom":
            custom = tool_call["custom"]
            return ResponseCustomToolCallParam(
                type="custom_tool_call",
                call_id=tool_call["id"],
                name=custom["name"],
                input=custom["input"],
            )
        case _:
            assert_never(tool_call["type"])


def _convert_message(
    message: ChatCompletionMessageParam,
) -> Generator[ResponseInputItemParam, None, None]:
    match message["role"]:
        case "user" | "assistant" | "system" | "developer":
            if message.get("function_call"):
                raise RequestValidationError(_DEPRECATED_FUNCTION_API)

            if tool_calls := message.get("tool_calls"):
                yield from map(_convert_tool_call, tool_calls)

            if (content := message.get("content")) is None:
                return

            role = message["role"]

            if isinstance(content, str):
                res_content = content
            else:
                res_content = [_convert_content_part(part) for part in content]

            yield EasyInputMessageParam(role=role, content=res_content)

        case "tool":
            content = message["content"]
            if isinstance(content, str):
                output = content
            else:
                output = [_convert_fun_call_part(part) for part in content]

            yield FunctionCallOutput(
                call_id=message["tool_call_id"],
                type="function_call_output",
                output=output,
            )

        case "function":
            raise RequestValidationError(_DEPRECATED_FUNCTION_API)

        case _:
            assert_never(message)


def convert_messages(
    messages: list[ChatCompletionMessageParam],
) -> ResponseInputParam:
    return [
        param for message in messages for param in _convert_message(message)
    ]


def _convert_output(output: list[ResponseOutputItem]) -> ChatCompletionMessage:
    text_content = ""
    web_search_calls_count = 0

    annotations: list[Annotation] = []
    attachments: list[Attachment] = []
    stages: list[Stage] = []
    tool_calls: list[ChatCompletionMessageToolCallUnion] = []

    for item in output:
        match item:
            case ResponseOutputMessage(content=content):
                for part in content:
                    match part:
                        case ResponseOutputText(text=text):
                            text_content += text
                            for annotation in part.annotations:
                                if res_annotation := convert_annotation(
                                    annotation
                                ):
                                    annotations.append(res_annotation)
                                    attachments.append(
                                        Attachment(
                                            title=res_annotation.url_citation.title,
                                            url=res_annotation.url_citation.url,
                                        )
                                    )
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

            case ResponseReasoningItem(summary=summary):
                if summary:
                    for index, summary_part in enumerate(summary):
                        suffix = "" if index == 0 else f" #{index + 1}"
                        stages.append(
                            Stage(
                                name="Reasoning" + suffix,
                                status=Status.COMPLETED,
                                content=summary_part.text,
                            )
                        )

            case ResponseFunctionWebSearch(id=item_id, action=action):
                logger.info(
                    f"[web_search] tool call: id={item_id}, action={action}"
                )
                web_search_calls_count += 1
                suffix = (
                    ""
                    if web_search_calls_count == 1
                    else f" #{web_search_calls_count}"
                )
                content = get_web_search_action_content(action)
                stages.append(
                    Stage(
                        name="Web Search" + suffix,
                        status=Status.COMPLETED,
                        content=content,
                    )
                )

            case (
                ResponseFileSearchToolCall()
                | ResponseComputerToolCall()
                | ResponseFunctionToolCallOutputItem()
                | ResponseComputerToolCallOutputItem()
                | ResponseToolSearchCall()
                | ResponseToolSearchOutputItem()
                | AdditionalTools()
                | ImageGenerationCall()
                | ResponseCodeInterpreterToolCall()
                | LocalShellCall()
                | LocalShellCallOutput()
                | McpCall()
                | McpListTools()
                | McpApprovalRequest()
                | McpApprovalResponse()
                | ResponseCompactionItem()
                | ResponseFunctionShellToolCall()
                | ResponseFunctionShellToolCallOutput()
                | ResponseApplyPatchToolCall()
                | ResponseApplyPatchToolCallOutput()
                | ResponseCustomToolCall()
                | ResponseCustomToolCallOutputItem()
            ):
                raise RequestValidationError(
                    f"The response output contains an unsupported item type: {item.type}"
                )
            case _:
                assert_never(item)

    extra_fields = {}
    if attachments or stages:
        extra_fields["custom_content"] = CustomContent(
            attachments=attachments or None,
            stages=stages or None,
        ).model_dump(mode="json", exclude_none=True)

    return ChatCompletionMessage(
        role="assistant",
        content=text_content,
        annotations=annotations or None,
        tool_calls=tool_calls or None,
        **extra_fields,
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
