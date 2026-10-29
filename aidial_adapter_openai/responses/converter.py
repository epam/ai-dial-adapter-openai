from collections.abc import Generator
from typing import Any, TypedDict, assert_never, cast

import pydantic
from aidial_sdk.chat_completion.enums import Status
from aidial_sdk.chat_completion.request import Attachment, CustomContent, Stage
from aidial_sdk.exceptions import RequestValidationError
from openai import Omit, omit
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
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
    ResponseOutputRefusalParam,
    ResponseOutputText,
    ResponseOutputTextParam,
    ResponseReasoningItem,
    ResponseTextConfigParam,
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
    Program,
    ProgramOutput,
)
from openai.types.responses.response_output_message_param import (
    Content as ResponseOutputContentParam,
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
from openai.types.shared_params import Reasoning

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.state import (
    MessageState,
    get_message_content_from_state,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.configuration import get_configuration
from aidial_adapter_openai.responses.request import convert_response_format
from aidial_adapter_openai.responses.response import (
    get_finish_reason,
    get_usage,
    get_web_search_action_content,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.validation import RequestValue

_NO_REFUSAL = "Refusal messages aren't yet supported."

_DEPRECATED_FUNCTION_API = "The deployment doesn't support the deprecated API for functions. Please use tools instead."


class _TokenizeResponsesRequest(TypedDict, total=False):
    model: str
    input: ResponseInputParam
    tools: list[Any] | Omit
    tool_choice: ToolChoice | Omit
    parallel_tool_calls: bool | Omit
    text: ResponseTextConfigParam | Omit


class _CreateResponsesRequest(TypedDict, total=False):
    model: str
    stream: bool
    input: ResponseInputParam
    tools: list[Any] | Omit
    tool_choice: ToolChoice | Omit
    parallel_tool_calls: bool | Omit
    text: ResponseTextConfigParam | Omit
    top_p: float | Omit
    temperature: float | Omit
    max_output_tokens: int | Omit
    reasoning: Reasoning | Omit


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


def convert_tool_choice(tool_choice: RequestValue) -> ToolChoice:
    if isinstance(tool_choice.unwrap(), str):
        return tool_choice.unwrap()

    tool_choice_type = tool_choice["type"].to_literal(
        "allowed_tools", "function", "custom"
    )
    match tool_choice_type:
        case "allowed_tools":
            allowed_tools = tool_choice["allowed_tools"]
            return ToolChoiceAllowedParam(
                type="allowed_tools",
                mode=allowed_tools["mode"].unwrap(),
                tools=allowed_tools["tools"].unwrap(),
            )
        case "function":
            return ToolChoiceFunctionParam(
                type="function",
                name=tool_choice["function"]["name"].unwrap(),
            )
        case "custom":
            return ToolChoiceCustomParam(
                type="custom", name=tool_choice["custom"]["name"].unwrap()
            )
        case _:
            assert_never(tool_choice_type)


def convert_tools(tools: RequestValue) -> list[ToolParam]:
    _allowed_static_function_names = {"web_search"}

    def _convert_tool(tool: RequestValue) -> ToolParam:
        match tool["type"].unwrap():
            case "static_function":
                static_function = tool.get("static_function")
                if static_function is None or not static_function.unwrap_dict():
                    raise RequestValidationError(
                        "Required field 'static_function' is empty or not found."
                    )

                static_function_name = static_function.get_str("name")
                if static_function_name not in _allowed_static_function_names:
                    msg = (
                        f"Provided static function name ('{static_function_name}') is not supported yet. "
                        f"Allowed values: {list(_allowed_static_function_names)}"
                    )
                    raise RequestValidationError(msg)

                configuration = static_function.get("configuration")
                return WebSearchToolParam(
                    type="web_search",
                    **(configuration.unwrap_dict() if configuration else {}),
                )
            case _:
                function = tool["function"]
                raw_function = function.unwrap_dict()
                return FunctionToolParam(
                    type="function",
                    name=function["name"].unwrap(),
                    parameters=raw_function.get("parameters"),
                    strict=raw_function.get("strict"),
                    description=raw_function.get("description"),
                )

    return [_convert_tool(tool) for tool in tools.to_list()]


def _convert_fun_call_part(
    part: RequestValue,
) -> ResponseFunctionCallOutputItemParam:
    part_type = part["type"].to_literal(
        "refusal", "text", "image_url", "file", "input_audio"
    )
    match part_type:
        case "refusal":
            raise RequestValidationError(_NO_REFUSAL)

        case "text":
            return ResponseInputTextContentParam(
                type="input_text", text=part["text"].unwrap()
            )

        case "image_url":
            image_url = part["image_url"]
            return ResponseInputImageContentParam(
                type="input_image",
                image_url=image_url["url"].unwrap(),
                detail=image_url.unwrap_dict().get("detail", "auto"),
            )

        case "file":
            file = part["file"].unwrap_dict()
            return ResponseInputFileContentParam(
                type="input_file",
                file_id=file.get("file_id"),
                file_data=file.get("file_data"),
                filename=file.get("filename"),
            )

        case "input_audio":
            raise RequestValidationError("Audio messages aren't supported")

        case _:
            assert_never(part_type)


def _to_output_content_param(part: RequestValue) -> ResponseOutputContentParam:
    part_type = part["type"].to_literal("refusal", "text")
    match part_type:
        case "refusal":
            return ResponseOutputRefusalParam(
                type="refusal", refusal=part["refusal"].unwrap()
            )
        case "text":
            return ResponseOutputTextParam(
                type="output_text", text=part["text"].unwrap(), annotations=[]
            )
        case _:
            assert_never(part_type)


def _to_input_content_param(part: RequestValue) -> ResponseInputContentParam:
    part_type = part["type"].to_literal(
        "text", "image_url", "file", "input_audio"
    )
    match part_type:
        case "text":
            return ResponseInputTextParam(
                type="input_text", text=part["text"].unwrap()
            )

        case "image_url":
            image_url = part["image_url"]
            return ResponseInputImageParam(
                type="input_image",
                image_url=image_url["url"].unwrap(),
                detail=image_url.unwrap_dict().get("detail", "auto"),
            )

        case "file":
            file = part["file"].unwrap_dict()
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
            assert_never(part_type)


def _convert_tool_call(
    tool_call: RequestValue,
) -> ResponseFunctionToolCallParam | ResponseCustomToolCallParam:
    call_id = tool_call["id"].unwrap()
    tool_call_type = tool_call["type"].to_literal("function", "custom")
    match tool_call_type:
        case "function":
            function = tool_call["function"]
            return ResponseFunctionToolCallParam(
                type="function_call",
                call_id=call_id,
                name=function["name"].unwrap(),
                arguments=function["arguments"].unwrap(),
            )
        case "custom":
            custom = tool_call["custom"]
            return ResponseCustomToolCallParam(
                type="custom_tool_call",
                call_id=call_id,
                name=custom["name"].unwrap(),
                input=custom["input"].unwrap(),
            )
        case _:
            assert_never(tool_call_type)


def _convert_message(
    idx: int, message: RequestValue
) -> Generator[ResponseInputItemParam, None, None]:
    role = message["role"].to_literal(
        "system", "developer", "user", "assistant", "tool", "function"
    )
    match role:
        case "system" | "developer" | "user":
            content = message["content"].to_str_or_list()
            if isinstance(content, str):
                res_content = content
            else:
                res_content = [
                    _to_input_content_param(part) for part in content
                ]

            yield EasyInputMessageParam(role=role, content=res_content)

        case "assistant":
            if state_content := get_message_content_from_state(
                idx, message.unwrap()
            ):
                yield from state_content

            if message.get("function_call") is not None:
                raise RequestValidationError(_DEPRECATED_FUNCTION_API)

            if (tool_calls := message.get("tool_calls")) is not None:
                yield from map(_convert_tool_call, tool_calls.to_list())

            if (content := message.get("content")) is None:
                return

            if isinstance(parts := content.to_str_or_list(), str):
                res_content = parts
            else:
                res_content = [_to_output_content_param(part) for part in parts]

            yield EasyInputMessageParam(role=role, content=res_content)  # type: ignore

        case "tool":
            content = message["content"].to_str_or_list()
            if isinstance(content, str):
                output = content
            else:
                output = [_convert_fun_call_part(part) for part in content]

            yield FunctionCallOutput(
                call_id=message["tool_call_id"].unwrap(),
                type="function_call_output",
                output=output,
            )

        case "function":
            raise RequestValidationError(_DEPRECATED_FUNCTION_API)

        case _:
            assert_never(role)


def convert_messages(messages: list[RequestValue]) -> ResponseInputParam:
    return [
        param
        for idx, message in enumerate(messages)
        for param in _convert_message(idx, message)
    ]


async def chat_completions_to_responses_request(
    request: RequestValue, file_storage: FileStorage | None
) -> tuple[_TokenizeResponsesRequest, _CreateResponsesRequest]:
    raw_request = request.unwrap_dict()
    is_stream = bool(request.get("stream"))
    # FIX_NOW: request["model"].to_str()
    model_name = raw_request["model"]
    messages = request["messages"].to_list()

    transformed_messages = await ResourceProcessor(
        file_storage=file_storage,
    ).transform_messages(messages)

    # The transformation keeps the order of messages,
    # so the transformed ones keep the paths of the original ones
    input_messages = convert_messages(
        [
            RequestValue(transformed.raw_message, message.path)
            for transformed, message in zip(
                transformed_messages, messages, strict=True
            )
        ]
    )

    res_tools = []
    if tools := request.get("tools"):
        res_tools.extend(convert_tools(tools))

    if (web_search_options := request.get("web_search_options")) is not None:
        res_tools.append(
            {"type": "web_search", **web_search_options.unwrap_dict()}
        )

    res_tool_choice: ToolChoice | Omit = omit
    if tool_choice := request.get("tool_choice"):
        res_tool_choice = convert_tool_choice(tool_choice)

    # FIX_NOW: use request
    max_output_tokens = (
        raw_request.get("max_tokens")
        or raw_request.get("max_completion_tokens")
        or omit
    )

    configuration = get_configuration(request)
    # FIX_NOW: use request
    parallel_tool_calls = (
        raw_request["parallel_tool_calls"]
        if raw_request.get("parallel_tool_calls") is not None
        else omit
    )

    text: ResponseTextConfigParam | Omit = omit
    if response_format := request.get("response_format"):
        text = ResponseTextConfigParam(
            format=convert_response_format(response_format)
        )

    tokenize_request: _TokenizeResponsesRequest = {
        "model": model_name,
        "input": input_messages,
        "tools": res_tools or omit,
        "tool_choice": res_tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "text": text,
    }
    create_request: _CreateResponsesRequest = {
        **tokenize_request,
        "stream": is_stream,
        # FIX_NOW: use request
        "top_p": raw_request.get("top_p") or omit,
        "temperature": raw_request.get("temperature") or omit,
        "max_output_tokens": max_output_tokens,
        "reasoning": configuration.reasoning or omit,
    }
    if extra := configuration.model_extra:
        # model_extra is dict[str, Any] pass-through fields;
        # cast lets them merge into the TypedDict, which is forwarded as-is to the API.
        create_request.update(cast(_CreateResponsesRequest, extra))

    return tokenize_request, create_request


def _convert_output(output: list[ResponseOutputItem]) -> ChatCompletionMessage:
    text_content = ""
    web_search_calls_count = 0

    annotations: list[Annotation] = []
    attachments: list[Attachment] = []
    stages: list[Stage] = []
    state: MessageState = MessageState(responses_output=[])
    tool_calls: list[ChatCompletionMessageToolCallUnion] = []
    reasoning_parts: list[str] = []

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
                    for summary_part in summary:
                        # The summary parts of all the reasoning items
                        # are numbered consecutively, just like
                        # in the streaming mode.
                        suffix = (
                            f" #{len(reasoning_parts) + 1}"
                            if reasoning_parts
                            else ""
                        )
                        reasoning_parts.append(summary_part.text)
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
                state.responses_output.append(item)

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
                | Program()
                | ProgramOutput()
            ):
                raise RequestValidationError(
                    f"The response output contains an unsupported item type: {item.type}"
                )
            case _:
                assert_never(item)

    extra_fields = {}
    if attachments or stages or state:
        extra_fields["custom_content"] = CustomContent(
            attachments=attachments or None,
            stages=stages or None,
            state=state.model_dump() or None,
        ).model_dump(mode="json", exclude_none=True)

    if reasoning_parts:
        extra_fields["reasoning_content"] = "\n\n".join(reasoning_parts)

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
