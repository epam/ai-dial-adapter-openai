import asyncio
from typing import Any, Dict, assert_never

import fastapi
from aidial_sdk.chat_completion import Choice, Stage
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import InternalServerError
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI, omit
from openai.types import Video

from aidial_adapter_openai.dial_api.attachment import create_dial_attachment
from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.video_generation.openai.configuration import (
    VideoGenerationConfig,
)
from aidial_adapter_openai.video_generation.openai.prompt import (
    VideoGenPrompt,
    get_last_file,
)
from aidial_adapter_openai.video_generation.request import validate_request


def _get_configuration(request: dict) -> VideoGenerationConfig:
    configuration = (
        parse_configuration(VideoGenerationConfig, request)
        or VideoGenerationConfig()
    )

    logger.debug(
        f"configuration: {configuration.model_dump_json(exclude_none=True)}"
    )
    return configuration


async def _poll_job(
    *,
    stage: Stage,
    client: AsyncOpenAI,
    video_job: Video,
    polling_interval: float,
) -> str:
    while True:
        status = video_job.status
        match status:
            case "completed":
                stage.append_content("Completed\n\n")
                return video_job.id

            case "failed":
                stage.append_content("Failed\n\n")
                message = "Video generation has failed"
                if err := video_job.error:
                    message += f": {err.message} (code={err.code})"
                raise InternalServerError(
                    message=message, display_message=message
                )

            case "queued":
                stage.append_content("Queued\n\n")

            case "in_progress":
                stage.append_content(f"In progress: {video_job.progress}%\n\n")

            case _:
                raise InternalServerError(f"Unexpected job status: {status}")
                assert_never(status)

        await asyncio.sleep(polling_interval)
        video_job = await client.videos.retrieve(video_job.id)


async def _download_video(
    *,
    response: DIALResponse,
    choice: Choice,
    client: AsyncOpenAI,
    video_id: str,
    storage: FileStorage | None,
):
    video_content = await client.videos.download_content(
        video_id, variant="video"
    )

    video_bytes = await video_content.aread()
    content_type = (
        video_content.response.headers.get("content-type") or "video/mp4"
    )

    choice.add_attachment(
        await create_dial_attachment(
            title="Video",
            content_type=content_type,
            data=video_bytes,
            file_storage=storage,
            upload_dir="videos",
        )
    )

    response.set_usage(prompt_tokens=0, completion_tokens=1)


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment_id: str,
    file_storage: FileStorage | None,
) -> StreamingResponse | dict:
    validate_request(request_body)

    if isinstance(client, AsyncAzureOpenAI):
        raise ValueError(
            "Only v1 API upstream endpoints are supported for OpenAI video generation deployments"
        )

    model_name = request_body["model"]
    configuration = _get_configuration(request_body)
    prompt = await VideoGenPrompt.from_request(request_body, file_storage)

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)

        with (
            response.create_single_choice() as choice,
            choice.create_stage(name="Video generation") as stage,
        ):
            video_job = await client.videos.create(
                model=model_name,
                prompt=prompt.prompt,
                input_reference=get_last_file(prompt) or omit,
                seconds=configuration.seconds or omit,  # type: ignore
                size=configuration.size or omit,  # type: ignore
                extra_body=configuration.model_extra,
            )

            video_id = await _poll_job(
                stage=stage,
                client=client,
                video_job=video_job,
                polling_interval=3.0,
            )

            await _download_video(
                response=response,
                choice=choice,
                storage=file_storage,
                client=client,
                video_id=video_id,
            )

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
