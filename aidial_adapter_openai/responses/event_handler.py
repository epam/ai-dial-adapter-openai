import json
import logging
from typing import Literal, assert_never

from aidial_sdk.exceptions import HTTPException as DialException
from openai import BaseModel
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import (
    Response,
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

from aidial_adapter_openai.responses.converter import convert_usage
from aidial_adapter_openai.utils.log_config import logger

ChatCompletionFinishReason = (
    Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    | None
)


class ErrorBody(BaseModel):
    message: str
    code: str | None = None
    param: str | None = None


class ErrorChunk(BaseModel):
    error: ErrorBody


class EventHandler:
    _id: str | None = None
    _created: int | None = None
    _model: str | None = None

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

    def _error(self, event: ResponseErrorEvent) -> ErrorChunk:
        return ErrorChunk(
            error=ErrorBody(
                message=event.message,
                code=event.code,
                param=event.param,
            )
        )

    def _usage(self, response: Response) -> CompletionUsage | None:
        if usage := response.usage:
            return convert_usage(usage)
        return None

    def _finish_reason(self, response: Response) -> ChatCompletionFinishReason:
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
        return "stop"

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
                        finish_reason=self._finish_reason(response),
                    ),
                    usage=self._usage(response),
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
                | ResponseFunctionCallArgumentsDeltaEvent()
                | ResponseFunctionCallArgumentsDoneEvent()
                | ResponseInProgressEvent()
                | ResponseFailedEvent()
                | ResponseOutputItemAddedEvent()
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
