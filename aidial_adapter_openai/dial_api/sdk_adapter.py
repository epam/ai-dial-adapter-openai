from typing import Callable, Coroutine

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.utils.streaming import to_block_response, to_streaming_response
from fastapi.responses import JSONResponse, StreamingResponse

from aidial_adapter_openai.dial_api.storage import DIAL_URL
from aidial_adapter_openai.exception_handlers import dial_exception_decorator


async def sdk_adapter(
    *,
    request: fastapi.Request,
    deployment_id: str,
    chat_completion: Callable[
        [DIALRequest, DIALResponse], Coroutine[None, None, None]
    ],
) -> fastapi.Response:
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
        content = await to_block_response(stream)
        return JSONResponse(content=content)
