import asyncio
import base64
from typing import Any, Dict, List, assert_never

import fastapi
from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.chat_completion import Stage
from aidial_sdk.exceptions import InternalServerError, RequestValidationError
from aidial_sdk.utils.streaming import to_block_response, to_streaming_response
from fastapi.responses import JSONResponse, StreamingResponse
from httpx._types import RequestFiles

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import DIAL_URL, FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.video_generation.azure.client import (
    AzureVideoAPIClient,
)
from aidial_adapter_openai.video_generation.azure.configuration import (
    VideoGenerationConfig,
)
from aidial_adapter_openai.video_generation.azure.prompt import VideoGenPrompt
from aidial_adapter_openai.video_generation.azure.types import (
    CreateVideoGenerationRequest,
    JobStatus,
    VideoGeneration,
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


def _get_configuration(request: dict) -> VideoGenerationConfig:
    configuration = (
        parse_configuration(VideoGenerationConfig, request)
        or VideoGenerationConfig()
    )

    logger.debug(f"configuration: {configuration.json(exclude_none=True)}")
    return configuration


async def _create_job(
    *,
    stage: Stage,
    client: AzureVideoAPIClient,
    request: CreateVideoGenerationRequest,
    files: RequestFiles,
) -> str:
    video_job = await client.create_job(request=request, files=files)
    stage.append_content(f"Status: {video_job.status}\n\n")
    return video_job.id


async def _poll_job(
    *,
    stage: Stage,
    client: AzureVideoAPIClient,
    job_id: str,
    polling_interval: float,
) -> List[VideoGeneration]:
    while True:
        await asyncio.sleep(polling_interval)

        video_job = await client.get_job_status(job_id)

        status = video_job.status
        stage.append_content(f"Status: {status}\n\n")

        match status:
            case JobStatus.SUCCEEDED:
                generations = video_job.generations or []
                if not generations:
                    raise InternalServerError(
                        "Video generation succeeded but no generations found"
                    )
                return generations

            case JobStatus.FAILED:
                video_job.failed()

            case JobStatus.CANCELLED:
                raise InternalServerError(
                    "Video generation job has been cancelled"
                )

            case (
                JobStatus.PREPROCESSING
                | JobStatus.QUEUED
                | JobStatus.RUNNING
                | JobStatus.PROCESSING
            ):
                continue
            case _:
                raise InternalServerError(f"Unexpected job status: {status}")
                assert_never(status)


async def _download_videos(
    *,
    response: DIALResponse,
    choice: Choice,
    client: AzureVideoAPIClient,
    video_generations: List[VideoGeneration],
    storage: FileStorage | None,
):
    n = len(video_generations)
    seconds = sum(v.n_seconds for v in video_generations)
    response.set_usage(
        prompt_tokens=0,
        completion_tokens=seconds,
    )

    for idx, video_generation in enumerate(video_generations, start=1):
        video_bytes = await client.get_video_content(video_generation.id)
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

        title = "video" if n == 1 else f"video #{idx}"

        choice.add_attachment(
            title=title, type=content_type, url=url, data=data
        )


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
    configuration = _get_configuration(request_body)
    prompt = await VideoGenPrompt.from_request(request_body, file_storage)
    inpaint_items, files = prompt.get_files()

    dial_request = await DIALRequest.from_request(
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
                    request=CreateVideoGenerationRequest.create(
                        model=model_name,
                        prompt=prompt.prompt,
                        width=configuration.width,
                        height=configuration.height,
                        n_seconds=configuration.n_seconds,
                        n_variants=configuration.n_variants,
                        inpaint_items=inpaint_items,
                    ),
                    files=files,
                    stage=stage,
                    client=client,
                )

                video_generations = await _poll_job(
                    stage=stage,
                    client=client,
                    job_id=job_id,
                    polling_interval=3.0,
                )

                await _download_videos(
                    response=response,
                    choice=choice,
                    storage=file_storage,
                    client=client,
                    video_generations=video_generations,
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
