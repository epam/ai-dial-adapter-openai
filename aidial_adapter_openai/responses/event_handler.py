import json
import logging
from typing import Dict, Generator, assert_never

import openai
import pydantic
from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCustomToolCallInputDeltaEvent,
    ResponseCustomToolCallInputDoneEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFileSearchCallInProgressEvent,
    ResponseFileSearchCallSearchingEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseImageGenCallCompletedEvent,
    ResponseImageGenCallGeneratingEvent,
    ResponseImageGenCallInProgressEvent,
    ResponseImageGenCallPartialImageEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseMcpCallArgumentsDeltaEvent,
    ResponseMcpCallArgumentsDoneEvent,
    ResponseMcpCallCompletedEvent,
    ResponseMcpCallFailedEvent,
    ResponseMcpCallInProgressEvent,
    ResponseMcpListToolsCompletedEvent,
    ResponseMcpListToolsFailedEvent,
    ResponseMcpListToolsInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseQueuedEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response_apply_patch_tool_call import (
    ResponseApplyPatchToolCall,
)
from openai.types.responses.response_apply_patch_tool_call_output import (
    ResponseApplyPatchToolCallOutput,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_compaction_item import (
    ResponseCompactionItem,
)
from openai.types.responses.response_computer_tool_call import (
    ResponseComputerToolCall,
)
from openai.types.responses.response_custom_tool_call import (
    ResponseCustomToolCall,
)
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
)
from openai.types.responses.response_function_shell_tool_call import (
    ResponseFunctionShellToolCall,
)
from openai.types.responses.response_function_shell_tool_call_output import (
    ResponseFunctionShellToolCallOutput,
)
from openai.types.responses.response_function_tool_call import (
    ResponseFunctionToolCall,
)
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from aidial_adapter_openai.responses.converter import (
    convert_annotation,
    parse_response_url_citation,
)
from aidial_adapter_openai.responses.response import (
    get_finish_reason,
    get_usage,
    get_web_search_action_content,
)
from aidial_adapter_openai.utils.log_config import logger


class ErrorBody(openai.BaseModel):
    message: str
    code: str | None = None
    param: str | None = None


class ErrorChunk(openai.BaseModel):
    error: ErrorBody


class EventHandler(pydantic.BaseModel):
    id_: str | None = None
    created_: int | None = None
    model_: str | None = None

    tool_calls: Dict[str, int] = {}
    """Map item_id for a tool call onto its index in the chat completion response"""

    stage_key_to_index: dict[str, int] = pydantic.Field(default_factory=dict)
    """Stable stage key to emitted stage index map."""

    stage_key_to_name: dict[str, str] = pydantic.Field(default_factory=dict)
    """Stable stage key to emitted stage display name map."""

    stage_base_name_count: dict[str, int] = pydantic.Field(default_factory=dict)
    """Per-stage-type counter for suffix generation."""

    @property
    def id(self) -> str:
        if self.id_ is None:
            raise DialException("Response ID is not set")
        return self.id_

    @property
    def created(self) -> int:
        if self.created_ is None:
            raise DialException("Response creation time is not set")
        return self.created_

    @property
    def model(self) -> str:
        if self.model_ is None:
            raise DialException("Response model is not set")
        return self.model_

    def _stage_chunk(self, stage: dict) -> ChatCompletionChunk:
        return self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(custom_content={"stages": [stage]}),  # type: ignore
            )
        )

    def _resolve_stage_index(self, stage_key: str) -> int:
        stage_index = self.stage_key_to_index.get(stage_key)
        if stage_index is None:
            raise DialException("Stage index not found.")
        return stage_index

    def _build_stage_name(self, base_name: str) -> str:
        """
        Appends suffix with counter to name, if stage is not the first one.
        """
        stage_count = self.stage_base_name_count.get(base_name, 0)
        self.stage_base_name_count[base_name] = stage_count + 1
        if stage_count == 0:
            return base_name
        suffix = f"#{stage_count + 1}"
        return f"{base_name} {suffix}"

    def _append_to_stage(
        self, stage_key: str, content: str
    ) -> ChatCompletionChunk:
        stage_index = self._resolve_stage_index(stage_key)
        stage: dict[str, int | str] = {"index": stage_index, "content": content}
        return self._stage_chunk(stage)

    def _open_stage(
        self, name: str, stage_key: str
    ) -> ChatCompletionChunk | None:
        """
        Returns None if the stage is already open, otherwise, a chunk opening the stage.
        """
        stage_index = self.stage_key_to_index.get(stage_key)
        if stage_index is not None:
            logger.info("Stage is already open. This step does nothing.")
            return None

        stage_index = len(self.stage_key_to_index)
        self.stage_key_to_index[stage_key] = stage_index
        self.stage_key_to_name[stage_key] = self._build_stage_name(name)
        name = self.stage_key_to_name[stage_key]
        stage = {"index": stage_index, "name": name}
        return self._stage_chunk(stage)

    def _close_stage(self, stage_key: str) -> ChatCompletionChunk:
        stage_index = self._resolve_stage_index(stage_key)
        stage: dict[str, int | str] = {
            "index": stage_index,
            "status": "completed",
        }
        return self._stage_chunk(stage)

    def _chunk(
        self,
        *,
        choice: Choice | None = None,
        usage: CompletionUsage | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=self.id,
            created=self.created,
            model=self.model,
            object="chat.completion.chunk",
            choices=[choice] if choice else [],
            usage=usage,
        )

    @staticmethod
    def _error(event: ResponseErrorEvent) -> ErrorChunk:
        return ErrorChunk(
            error=ErrorBody(
                message=event.message,
                code=event.code,
                param=event.param,
            )
        )

    def _tool_call_chunk_open(
        self,
        item_id: str,
        arguments: str,
        name: str,
        call_id: str,
    ) -> ChatCompletionChunk:
        idx = len(self.tool_calls)
        self.tool_calls[item_id] = idx

        return self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=idx,
                            id=call_id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                arguments=arguments, name=name
                            ),
                        )
                    ]
                ),
            )
        )

    def _tool_call_delta(self, item_id: str, delta: str) -> ChatCompletionChunk:
        if (idx := self.tool_calls.get(item_id)) is None:
            raise InternalServerError(
                "Cannot add delta to an unopened tool call"
            )

        return self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=idx,
                            function=ChoiceDeltaToolCallFunction(
                                arguments=delta
                            ),
                        )
                    ]
                ),
            )
        )

    def _annotation_chunks(
        self, annotation: dict
    ) -> Generator[ChatCompletionChunk, None, None]:
        parsed_annotation = parse_response_url_citation(annotation)
        if parsed_annotation is None:
            return

        converted_annotation = convert_annotation(parsed_annotation)
        if converted_annotation is None:
            return

        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    annotations=[converted_annotation]  # type: ignore
                ),
            )
        )

        attachment = Attachment(
            title=converted_annotation.url_citation.title,
            url=converted_annotation.url_citation.url,
        )
        attachment_dict = attachment.model_dump(mode="json", exclude_none=True)
        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    custom_content={  # type: ignore
                        "attachments": [attachment_dict]
                    }
                ),
            )
        )

    def handle(
        self, event: ResponseStreamEvent
    ) -> Generator[ChatCompletionChunk | ErrorChunk, None, None]:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"event[{event.type}]: {json.dumps(event.model_dump())}"
            )

        match event:
            case ResponseCreatedEvent(response=response):
                self.id_ = response.id
                self.created_ = int(response.created_at)
                self.model_ = response.model
                yield self._chunk(
                    choice=Choice(index=0, delta=ChoiceDelta(role="assistant"))
                )

            case ResponseErrorEvent():
                yield self._error(event)

            case ResponseTextDeltaEvent(delta=delta):
                yield self._chunk(
                    choice=Choice(index=0, delta=ChoiceDelta(content=delta))
                )

            case (
                ResponseCompletedEvent(response=response)
                | ResponseIncompleteEvent(response=response)
            ):
                yield self._chunk(
                    choice=Choice(
                        index=0,
                        delta=ChoiceDelta(content=""),
                        finish_reason=get_finish_reason(response),
                    ),
                    usage=get_usage(response),
                )

            case ResponseOutputItemDoneEvent(item=item):
                match item:
                    case ResponseFunctionWebSearch(action=action, id=stage_id):
                        content = get_web_search_action_content(action)
                        stage_key = f"web_search:{stage_id}"
                        yield self._append_to_stage(
                            stage_key=stage_key, content=content
                        )
                        yield self._close_stage(stage_key=stage_key)

            case ResponseWebSearchCallInProgressEvent(item_id=stage_id):
                stage_key = f"web_search:{stage_id}"
                chunk = self._open_stage(name="Web Search", stage_key=stage_key)
                if chunk is not None:
                    yield chunk

            case ResponseOutputItemAddedEvent(item=item):
                match item:
                    case ResponseFunctionToolCall(
                        arguments=arguments,
                        name=name,
                        call_id=call_id,
                        id=item_id,
                    ):
                        if item_id is None:
                            raise InternalServerError(
                                "item_id of a tool call is missing"
                            )
                        yield self._tool_call_chunk_open(
                            item_id, arguments, name, call_id
                        )

                    case (
                        ResponseOutputMessage()
                        | ResponseFileSearchToolCall()
                        | ResponseFunctionToolCall()
                        | ResponseFunctionWebSearch()
                        | ResponseComputerToolCall()
                        | ResponseReasoningItem()
                        | ResponseCodeInterpreterToolCall()
                        | ImageGenerationCall()
                        | LocalShellCall()
                        | McpCall()
                        | McpListTools()
                        | McpApprovalRequest()
                        | ResponseCompactionItem()
                        | ResponseFunctionShellToolCall()
                        | ResponseFunctionShellToolCallOutput()
                        | ResponseApplyPatchToolCall()
                        | ResponseApplyPatchToolCallOutput()
                        | ResponseCustomToolCall()
                    ):
                        pass
                    case _:
                        assert_never(item)

            case ResponseFunctionCallArgumentsDeltaEvent(
                delta=delta, item_id=item_id
            ):
                yield self._tool_call_delta(item_id, delta)

            case ResponseReasoningSummaryPartAddedEvent(
                item_id=stage_id, summary_index=summary_index
            ):
                stage_key = f"reasoning:{stage_id}:{summary_index}"
                chunk = self._open_stage(name="Reasoning", stage_key=stage_key)
                if chunk is not None:
                    yield chunk

            case ResponseReasoningSummaryTextDeltaEvent(
                item_id=stage_id, summary_index=summary_index, delta=content
            ):
                stage_key = f"reasoning:{stage_id}:{summary_index}"
                yield self._append_to_stage(
                    stage_key=stage_key, content=content
                )

            case ResponseReasoningSummaryTextDoneEvent(
                item_id=stage_id, summary_index=summary_index
            ):
                stage_key = f"reasoning:{stage_id}:{summary_index}"
                yield self._close_stage(stage_key=stage_key)

            case ResponseOutputTextAnnotationAddedEvent(annotation=annotation):
                if isinstance(annotation, dict):
                    yield from self._annotation_chunks(annotation)
                else:
                    logger.warning(
                        f"Unsupported annotation payload type in stream: {type(annotation)}"
                    )

            case (
                ResponseAudioDeltaEvent()
                | ResponseAudioDoneEvent()
                | ResponseAudioTranscriptDeltaEvent()
                | ResponseAudioTranscriptDoneEvent()
                | ResponseCodeInterpreterCallCodeDeltaEvent()
                | ResponseCodeInterpreterCallCodeDoneEvent()
                | ResponseCodeInterpreterCallCompletedEvent()
                | ResponseCodeInterpreterCallInProgressEvent()
                | ResponseCodeInterpreterCallInterpretingEvent()
                | ResponseContentPartAddedEvent()
                | ResponseContentPartDoneEvent()
                | ResponseFileSearchCallCompletedEvent()
                | ResponseFileSearchCallInProgressEvent()
                | ResponseFileSearchCallSearchingEvent()
                | ResponseFunctionCallArgumentsDoneEvent()
                | ResponseInProgressEvent()
                | ResponseFailedEvent()
                | ResponseReasoningSummaryPartDoneEvent()
                | ResponseRefusalDeltaEvent()
                | ResponseRefusalDoneEvent()
                | ResponseTextDoneEvent()
                | ResponseWebSearchCallSearchingEvent()
                | ResponseWebSearchCallCompletedEvent()
                | ResponseImageGenCallCompletedEvent()
                | ResponseImageGenCallGeneratingEvent()
                | ResponseImageGenCallInProgressEvent()
                | ResponseImageGenCallPartialImageEvent()
                | ResponseMcpCallArgumentsDeltaEvent()
                | ResponseMcpCallArgumentsDoneEvent()
                | ResponseMcpCallCompletedEvent()
                | ResponseMcpCallFailedEvent()
                | ResponseMcpCallInProgressEvent()
                | ResponseMcpListToolsCompletedEvent()
                | ResponseMcpListToolsFailedEvent()
                | ResponseMcpListToolsInProgressEvent()
                | ResponseQueuedEvent()
                | ResponseReasoningSummaryPartDoneEvent()
                | ResponseReasoningTextDeltaEvent()
                | ResponseReasoningTextDoneEvent()
                | ResponseCustomToolCallInputDeltaEvent()
                | ResponseCustomToolCallInputDoneEvent()
            ):
                pass
            case _:
                assert_never(event)
