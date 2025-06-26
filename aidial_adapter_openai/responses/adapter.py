import json
import logging
from typing import Any, AsyncIterator, Dict, List

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import Response as FastAPIResponse
from openai import NOT_GIVEN, AsyncStream, BaseModel

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.gpt4_multi_modal.chat_completion import USAGE
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.responses.converter import (
    convert_messages,
    convert_response,
)
from aidial_adapter_openai.responses.event_handler import EventHandler
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)
from aidial_adapter_openai.utils.streaming import (
    create_response_from_chunk,
    create_stage_chunk,
    map_stream,
)


async def chat_completion(
    request: Dict[str, Any],
    endpoint: OpenAIEndpoint | AzureOpenAIEndpoint,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: FileStorage | None,
    api_version: str,
    deployment: str,
) -> AsyncIterator[dict] | dict | FastAPIResponse:

    if (n_param := request.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n_param}."
        )

    client = endpoint.get_client({**creds, "api_version": api_version})

    messages: List[Any] = request["messages"]
    if len(messages) == 0:
        raise RequestValidationError("The request doesn't contain any messages")

    transform_result = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)

    if isinstance(transform_result, DialException):
        logger.error(f"Failed to prepare request: {transform_result.message}")
        chunk = create_stage_chunk("Usage", USAGE, is_stream)
        return create_response_from_chunk(chunk, transform_result, is_stream)

    input_messages = convert_messages(
        [m.raw_message for m in transform_result]  # type: ignore
    )

    response = await client.responses.create(
        model=deployment,
        stream=is_stream,
        input=input_messages,
        tools=request.get("tools") or NOT_GIVEN,
        tool_choice=request.get("tool_choice") or NOT_GIVEN,
        top_p=request.get("top_p") or NOT_GIVEN,
        temperature=request.get("temperature") or NOT_GIVEN,
        max_output_tokens=request.get("max_tokens") or NOT_GIVEN,
        parallel_tool_calls=request.get("parallel_tool_calls") or NOT_GIVEN,
    )

    def _to_dict(x: BaseModel):
        ret = x.dict()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"response: {json.dumps(ret)}")
        return ret

    if isinstance(response, AsyncStream):
        handler = EventHandler()
        return map_stream(_to_dict, map_stream(handler.handle, response))
    else:
        return _to_dict(convert_response(response))
