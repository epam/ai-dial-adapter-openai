import asyncio
from typing import Any, Dict, assert_never

import fastapi
from aidial_sdk.chat_completion import Choice, Stage
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import InternalServerError, InvalidRequestError
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, omit
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


def _parse_configuration(request: dict) -> VideoGenerationConfig:
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

                message = "Video generation job failed"
                code = None
                if err := video_job.error:
                    message += f": {err.message}"
                    code = err.code

                if code == "moderation_blocked":
                    code = "content_filter"
                    raise InvalidRequestError(message=message, code=code)
                else:
                    raise InternalServerError(message=message, code=code)

            case "queued":
                stage.append_content("Queued\n\n")

            case "in_progress":
                stage.append_content("In progress\n\n")

            case _:
                raise InternalServerError(f"Unexpected job status: {status}")
                assert_never(status)

        await asyncio.sleep(polling_interval)
        video_job = await client.videos.retrieve(video_job.id)


async def _download_video(
    *,
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


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Dict[str, Any],
    client: AsyncOpenAI,
    deployment_id: str,
    file_storage: FileStorage | None,
) -> StreamingResponse | dict:
    validate_request(request_body)

    model_name = request_body["model"]
    configuration = _parse_configuration(request_body)
    prompt = await VideoGenPrompt.from_request(request_body, file_storage)

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)

        with (
            response.create_single_choice() as choice,
            choice.create_stage(name="Video generation") as stage,
        ):
            if (seconds_arg := configuration.seconds) is None:
                seconds = omit
            else:
                seconds = str(seconds_arg)

            video_job = await client.videos.create(
                model=model_name,
                prompt=prompt.prompt,
                input_reference=get_last_file(prompt) or omit,
                seconds=seconds,  # type: ignore
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
                choice=choice,
                storage=file_storage,
                client=client,
                video_id=video_id,
            )

            response.set_usage(
                prompt_tokens=0, completion_tokens=int(video_job.seconds)
            )

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
