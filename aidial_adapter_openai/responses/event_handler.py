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
from openai.types.chat.chat_completion_message import (
    Annotation,
    AnnotationURLCitation,
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

from aidial_adapter_openai.responses.response import (
    get_finish_reason,
    get_usage,
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
    """Map item_id for a tool call onto its index in the chat completion response
    """

    stage: dict = {}
    """Stage life cycle memory"""

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

    def _append_to_stage(
        self, content: str, name: str | None = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        if not self.stage:
            yield self._open_stage(name=name or "Default")

        self.stage["index"] += 1
        if name:
            self.stage["name"] = name

        suffix = f"#{self.stage['index']}"
        name = f"{self.stage['name']} {suffix}"
        stage = {"index": self.stage["index"], "name": name, "content": content}
        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    custom_content={  # type: ignore
                        "stages": [stage]
                    }
                ),
            )
        )

    def _open_stage(
        self, name: str, stage_index: int = 0
    ) -> ChatCompletionChunk:
        self.stage = {"index": stage_index, "name": name}
        return self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    custom_content={  # type: ignore
                        "stages": [self.stage]
                    }
                ),
            )
        )

    def _close_stage(
        self, name: str | None = None
    ) -> Generator[ChatCompletionChunk, None, None]:
        if not self.stage:
            yield self._open_stage(name=name or "Default")

        self.stage["index"] += 1
        self.stage["status"] = "completed"
        stage = {"index": self.stage["index"], "status": self.stage["status"]}
        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    custom_content={"stages": [stage]}  # type: ignore
                ),
            )
        )

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
        self, annotation: dict[str, object]
    ) -> Generator[ChatCompletionChunk, None, None]:
        annotation_type = annotation.get("type")
        if annotation_type != "url_citation":
            logger.warning(
                f"Unsupported annotation payload type in stream: {annotation_type}"
            )
            return

        start_index = annotation.get("start_index")
        end_index = annotation.get("end_index")
        title = annotation.get("title")
        url = annotation.get("url")
        if not isinstance(start_index, int) or not isinstance(end_index, int):
            logger.warning("Annotation citation indices are invalid")
            return
        if not isinstance(title, str) or not isinstance(url, str):
            logger.warning("Annotation citation title/url are invalid")
            return

        annotation_ = Annotation(
            type="url_citation",
            url_citation=AnnotationURLCitation(
                start_index=start_index,
                end_index=end_index,
                title=title,
                url=url,
            ),
        )
        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    annotations=[annotation_]  # type: ignore
                ),
            )
        )

        attachment = Attachment(
            title=title,
            url=url,
        ).model_dump(mode="json", exclude_none=True)
        yield self._chunk(
            choice=Choice(
                index=0,
                delta=ChoiceDelta(
                    custom_content={  # type: ignore
                        "attachments": [attachment]
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
                    choice=Choice(
                        index=0,
                        delta=ChoiceDelta(content=delta, role="assistant"),
                    )
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
                    case ResponseFunctionWebSearch(action=action):
                        action_dump = action.model_dump()
                        query = action_dump.get("query", "")
                        action_type = action_dump.get("type", "unknown")
                        details = [f"type: {action_type}"]
                        if query:
                            details.append(f"query: {query}")
                        yield from self._append_to_stage(
                            content=json.dumps("\n".join(details)),
                        )

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
                    case ResponseFunctionWebSearch():
                        yield self._open_stage("Web Search")
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

            case ResponseReasoningSummaryTextDeltaEvent(delta=content):
                yield from self._append_to_stage(
                    content=content, name="Reasoning"
                )

            case ResponseReasoningSummaryTextDoneEvent():
                yield from self._close_stage()

            case ResponseWebSearchCallCompletedEvent():
                yield from self._close_stage()

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
                | ResponseOutputItemDoneEvent()
                | ResponseReasoningSummaryPartAddedEvent()
                | ResponseReasoningSummaryPartDoneEvent()
                | ResponseRefusalDeltaEvent()
                | ResponseRefusalDoneEvent()
                | ResponseTextDoneEvent()
                | ResponseWebSearchCallInProgressEvent()
                | ResponseWebSearchCallSearchingEvent()
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
                | ResponseReasoningSummaryPartAddedEvent()
                | ResponseReasoningSummaryPartDoneEvent()
                | ResponseReasoningSummaryTextDeltaEvent()
                | ResponseReasoningSummaryTextDoneEvent()
                | ResponseReasoningTextDeltaEvent()
                | ResponseReasoningTextDoneEvent()
                | ResponseCustomToolCallInputDeltaEvent()
                | ResponseCustomToolCallInputDoneEvent()
            ):
                pass
            case _:
                assert_never(event)
