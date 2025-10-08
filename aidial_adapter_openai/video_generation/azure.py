import asyncio
import base64
import json
from typing import Any, Dict, List

import fastapi
import httpx
from aidial_sdk.chat_completion import Choice
from aidial_sdk.chat_completion import Request
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.chat_completion import Stage
from aidial_sdk.exceptions import InternalServerError, RequestValidationError
from aidial_sdk.utils.streaming import to_block_response, to_streaming_response
from fastapi.responses import JSONResponse, StreamingResponse

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import DIAL_URL, FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class VideoGenerationConfig(ExtraAllowedModel):
    width: int = 480
    height: int = 480
    n_seconds: int = 5
    n_variants: int = 1


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


def _user_facing_error(
    message: str, response: httpx.Response | None = None
) -> InternalServerError:
    if response:
        logger.error(f"{message}: {response.status_code} {response.text}")
    else:
        logger.error(message)
    return InternalServerError(message=message, display_message=message)


async def _create_job(
    *, stage: Stage, endpoint: str, headers: dict, body: Dict[str, Any]
) -> str:
    resp = await get_http_client().post(
        url=f"{endpoint}/jobs",
        json=body,
        headers=headers,
        params={"api-version": "preview"},
    )

    if not resp.is_success:
        raise _user_facing_error("Video generation job creation failed", resp)

    resp_body = resp.json()

    if status := resp_body.get("status"):
        stage.append_content(f"Status: {status}\n\n")

    return resp_body["id"]


async def _poll_job(
    *,
    response: DIALResponse,
    choice: Choice,
    stage: Stage,
    storage: FileStorage | None,
    endpoint: str,
    headers: dict,
    job_id: str,
    polling_interval: float,
) -> None:
    job_url = f"{endpoint}/jobs/{job_id}"

    while True:
        await asyncio.sleep(polling_interval)

        job_status_response = await get_http_client().get(
            url=job_url,
            headers=headers,
            params={"api-version": "preview"},
        )

        if not job_status_response.is_success:
            raise _user_facing_error(
                "Polling video generation job status failed",
                job_status_response,
            )

        job_info = job_status_response.json()

        logger.debug(f"job status: {json.dumps(job_info)}")

        status = job_info.get("status")

        if status:
            stage.append_content(f"Status: {status}\n\n")

        if status == "succeeded":
            generations = job_info.get("generations") or []
            if not generations:
                raise _user_facing_error(
                    "Video generation succeeded but no generations found"
                )

            response.set_usage(
                prompt_tokens=0, completion_tokens=len(generations)
            )

            for idx, generation in enumerate(generations, start=1):
                title = "video"
                if len(generations) > 1:
                    title += f" #{idx}"

                generation_id = generation.get("id")
                video_url = f"{endpoint}/{generation_id}/content/video"
                video_response = await get_http_client().get(
                    url=video_url,
                    headers=headers,
                    params={"api-version": "preview"},
                )
                if not video_response.is_success:
                    raise _user_facing_error(
                        "Fetching generated video content failed",
                        video_response,
                    )

                video_bytes = video_response.content
                content_type = "video/mp4"

                if storage:
                    file_metadata = await storage.upload_file(
                        "videos", video_bytes, content_type
                    )
                    url = file_metadata["url"]
                    data = None
                else:
                    url = None
                    data = base64.b64encode(video_bytes).decode("utf-8")

                choice.add_attachment(
                    title=title, type=content_type, url=url, data=data
                )

            return

        if status == "failed":
            message = "Video generation job failed"
            if reason := job_info.get("failure_reason"):
                message += f": {reason}"
            raise _user_facing_error(message)


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


def _get_headers(creds: OpenAICreds) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if key := creds.get("api_key"):
        headers["api_key"] = key
    if token := creds.get("azure_ad_token"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


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

    headers = _get_headers(creds)

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)

        with response.create_single_choice() as choice:
            with choice.create_stage(name="Generation") as stage:
                job_id = await _create_job(
                    body={
                        "model": model_name,
                        "prompt": prompt,
                        **configuration.dict(),
                    },
                    stage=stage,
                    headers=headers,
                    endpoint=upstream_endpoint,
                )

                await _poll_job(
                    response=response,
                    choice=choice,
                    stage=stage,
                    job_id=job_id,
                    headers=headers,
                    storage=file_storage,
                    endpoint=upstream_endpoint,
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
