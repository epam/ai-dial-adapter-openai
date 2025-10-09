import asyncio
import base64
import json
from typing import Any, Dict, List

import fastapi
from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion import Request
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.chat_completion import Stage
from aidial_sdk.exceptions import RequestValidationError
from aidial_sdk.utils.streaming import to_block_response, to_streaming_response
from fastapi.responses import JSONResponse, StreamingResponse

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import DIAL_URL, FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.video_generation.azure.client import (
    AzureVideoAPIClient,
    JobStatus,
    user_facing_error,
)
from aidial_adapter_openai.video_generation.azure.configuration import (
    VideoGenerationConfig,
)


def _validate_request(request: Dict[str, Any]) -> None:
    errors: List[str] = []

    if (n := request.get("n")) not in [None, 1]:
        errors.append(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    unsupported_params: List[str] = []
    for param in [
        "stop",
        "seed",
        "top_logprobs",
        "logprobs",
        "presence_penalty",
        "function_call",
        "functions",
        "tools",
        "tool_choice",
    ]:
        if request.get(param) is not None:
            unsupported_params.append(param)

    if unsupported_params:
        suffix = "s" if len(unsupported_params) > 1 else ""
        errors.append(
            f"The deployment doesn't support {', '.join(unsupported_params)} request parameter{suffix}."
        )

    if not request.get("messages"):
        errors.append("The request doesn't contain any messages.")

    if errors:
        raise RequestValidationError(" ".join(errors))


def _get_configuration(request: Dict[str, Any]) -> VideoGenerationConfig:
    configuration = (
        parse_configuration(VideoGenerationConfig, request)
        or VideoGenerationConfig()
    )

    logger.debug(f"configuration: {configuration.json()}")
    return configuration


def _get_prompt(request_body: Dict[str, Any]) -> str:
    messages = request_body["messages"]
    prompt = messages[-1].get("content") or ""
    if not isinstance(prompt, str):
        raise RequestValidationError(
            "The last message must contain a text content."
        )
    if not prompt.strip():
        raise RequestValidationError("The prompt cannot be empty.")
    return prompt


async def _create_job(
    *, stage: Stage, client: AzureVideoAPIClient, request: Dict[str, Any]
) -> str:
    resp = await client.post_job(request=request)
    stage.append_content(f"Status: {resp.status}\n\n")
    return resp.id


async def _poll_job(
    *,
    response: DIALResponse,
    choice: Choice,
    stage: Stage,
    storage: FileStorage | None,
    client: AzureVideoAPIClient,
    job_id: str,
    polling_interval: float,
) -> None:
    while True:
        await asyncio.sleep(polling_interval)

        job_info = await client.get_job_status(job_id)
        logger.debug(f"job info: {json.dumps(job_info.dict())}")

        status = job_info.status

        if status:
            stage.append_content(f"Status: {status}\n\n")

        if status == JobStatus.SUCCEEDED:
            generations = job_info.generations or []
            if not generations:
                raise user_facing_error(
                    "Video generation succeeded but no generations found"
                )

            response.set_usage(
                prompt_tokens=0, completion_tokens=len(generations)
            )

            for idx, generation in enumerate(generations, start=1):
                video_bytes = await client.get_video_content(generation.id)
                content_type = "video/mp4"

                if storage:
                    metadata = await storage.upload_file(
                        "videos", video_bytes, content_type
                    )
                    url = metadata["url"]
                    data = None
                else:
                    url = None
                    data = base64.b64encode(video_bytes).decode("utf-8")

                title = "video"
                if len(generations) > 1:
                    title += f" #{idx}"

                choice.add_attachment(
                    title=title, type=content_type, url=url, data=data
                )

            return


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Dict[str, Any],
    creds: OpenAICreds,
    deployment_id: str,
    upstream_endpoint: str,
    file_storage: FileStorage | None,
) -> fastapi.Response:
    _validate_request(request_body)

    model_name = request_body["model"]
    prompt = _get_prompt(request_body)
    configuration = _get_configuration(request_body)

    dial_request = await Request.from_request(
        request=request,
        deployment_id=deployment_id,
        base_url=DIAL_URL,
    )
    response = DIALResponse(request=dial_request)

    client = AzureVideoAPIClient(creds=creds, base_url=upstream_endpoint)

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)

        with response.create_single_choice() as choice:
            with choice.create_stage(name="Generation") as stage:
                job_id = await _create_job(
                    request={
                        "model": model_name,
                        "prompt": prompt,
                        **configuration.dict(),
                    },
                    stage=stage,
                    client=client,
                )

                await _poll_job(
                    response=response,
                    choice=choice,
                    stage=stage,
                    job_id=job_id,
                    storage=file_storage,
                    client=client,
                    polling_interval=3.0,
                )

    stream = response._generate_stream(_handler)

    if dial_request.stream:
        return StreamingResponse(
            await to_streaming_response(stream),
            media_type="text/event-stream",
        )
    else:
        content = await to_block_response(stream)
        return JSONResponse(content=content)
