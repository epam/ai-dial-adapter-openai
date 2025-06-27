import json
import logging
from typing import Dict, assert_never

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from openai import BaseModel
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
    ResponseReasoningDeltaEvent,
    ResponseReasoningDoneEvent,
    ResponseReasoningSummaryDeltaEvent,
    ResponseReasoningSummaryDoneEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_computer_tool_call import (
    ResponseComputerToolCall,
)
from openai.types.responses.response_file_search_tool_call import (
    ResponseFileSearchToolCall,
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


class ErrorBody(BaseModel):
    message: str
    code: str | None = None
    param: str | None = None


class ErrorChunk(BaseModel):
    error: ErrorBody


class EventHandler(BaseModel):
    _id: str | None = None
    _created: int | None = None
    _model: str | None = None

    _tool_calls: Dict[str, int] = {}
    """Map item_id for a tool call onto its index in the chat completion response
    """

    @property
    def id(self) -> str:
        if self._id is None:
            raise DialException("Response ID is not set")
        return self._id

    @property
    def created(self) -> int:
        if self._created is None:
            raise DialException("Response creation time is not set")
        return self._created

    @property
    def model(self) -> str:
        if self._model is None:
            raise DialException("Response model is not set")
        return self._model

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
        idx = len(self._tool_calls)
        self._tool_calls[item_id] = idx

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
        if (idx := self._tool_calls.get(item_id)) is None:
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

    def handle(
        self, event: ResponseStreamEvent
    ) -> ChatCompletionChunk | ErrorChunk | None:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"event: {json.dumps(event.dict())}")

        match event:
            case ResponseCreatedEvent(response=response):
                self._id = response.id
                self._created = int(response.created_at)
                self._model = response.model
                return self._chunk(
                    choice=Choice(index=0, delta=ChoiceDelta(role="assistant"))
                )

            case ResponseErrorEvent():
                return self._error(event)

            case ResponseTextDeltaEvent(delta=delta):
                return self._chunk(
                    choice=Choice(index=0, delta=ChoiceDelta(content=delta))
                )

            case ResponseCompletedEvent(
                response=response
            ) | ResponseIncompleteEvent(response=response):
                return self._chunk(
                    choice=Choice(
                        index=0,
                        delta=ChoiceDelta(content=""),
                        finish_reason=get_finish_reason(response),
                    ),
                    usage=get_usage(response),
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
                        return self._tool_call_chunk_open(
                            item_id, arguments, name, call_id
                        )

                    case (
                        ResponseOutputMessage()
                        | ResponseFileSearchToolCall()
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
                        pass
                    case _:
                        assert_never(item)

            case ResponseFunctionCallArgumentsDeltaEvent(
                delta=delta, item_id=item_id
            ):
                return self._tool_call_delta(item_id, delta)

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
                | ResponseReasoningSummaryTextDeltaEvent()
                | ResponseReasoningSummaryTextDoneEvent()
                | ResponseRefusalDeltaEvent()
                | ResponseRefusalDoneEvent()
                | ResponseTextDoneEvent()
                | ResponseWebSearchCallCompletedEvent()
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
                | ResponseOutputTextAnnotationAddedEvent()
                | ResponseQueuedEvent()
                | ResponseReasoningDeltaEvent()
                | ResponseReasoningDoneEvent()
                | ResponseReasoningSummaryDeltaEvent()
                | ResponseReasoningSummaryDoneEvent()
            ):
                return None
            case _:
                assert_never(event)
