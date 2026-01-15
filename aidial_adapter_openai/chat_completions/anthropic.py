from typing import Any

import fastapi
from anthropic import AsyncAnthropicFoundry
from fastapi.responses import StreamingResponse

from aidial_adapter_openai.dial_api.storage import FileStorage


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Any,
    deployment_id: str,
    client: AsyncAnthropicFoundry,
    file_storage: FileStorage | None,
) -> StreamingResponse | dict:
    return {}
