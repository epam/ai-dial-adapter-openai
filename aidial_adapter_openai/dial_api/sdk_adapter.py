from collections.abc import Callable, Coroutine

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.utils.streaming import to_block_response, to_streaming_response
from fastapi.responses import StreamingResponse

from aidial_adapter_openai.dial_api.storage import DIAL_URL
from aidial_adapter_openai.exceptions.handlers import dial_exception_decorator
from aidial_adapter_openai.utils.validation import RequestValue


async def sdk_adapter(
    *,
    request: fastapi.Request,
    deployment_id: str,
    chat_completion: Callable[
        [DIALRequest, DIALResponse], Coroutine[None, None, None]
    ],
) -> StreamingResponse | dict:
    # Parsed upfront to report an invalid request body with the exact path,
    # since the DIAL SDK raises a bare pydantic error on it
    # FIX_NOW: do we really need it? How did it work before? Indeed SDK should have
    # report the parsing validation error and I'm fine with it.
    RequestValue(await request.json()).to_model(ChatCompletionRequest)

    dial_request = await DIALRequest.from_request(
        request=request,
        deployment_id=deployment_id,
        base_url=DIAL_URL,
    )

    response = DIALResponse(request=dial_request)

    @dial_exception_decorator
    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        await chat_completion(request, response)

    stream = response._generate_stream(_handler)

    if dial_request.stream:
        return StreamingResponse(
            await to_streaming_response(stream),
            media_type="text/event-stream",
        )
    else:
        return await to_block_response(stream)
