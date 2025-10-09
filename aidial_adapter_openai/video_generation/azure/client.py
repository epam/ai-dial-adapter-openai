from enum import Enum
from typing import Dict, List, Literal, NoReturn, Self

import httpx
from aidial_sdk.exceptions import InternalServerError
from pydantic import BaseModel

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger


class JobStatus(str, Enum):
    PREPROCESSING = "preprocessing"
    QUEUED = "queued"
    RUNNING = "running"
    PROCESSING = "processing"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VideoGeneration(BaseModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/aae85aa3e7e4fda95ea2d3abac0ba1d8159db214/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L16081
    """

    id: str


class VideoGenerationJob(BaseModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/aae85aa3e7e4fda95ea2d3abac0ba1d8159db214/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L16123
    """

    id: str
    status: JobStatus
    generations: List[VideoGeneration] | None = None
    failure_reason: (
        str | Literal["input_moderation", "input_moderation"] | None
    ) = None

    def raise_on_failure(self) -> Self:
        if self.status == JobStatus.FAILED:
            self.failed()
        return self

    def failed(self) -> NoReturn:
        message = "Video generation job failed"
        if reason := self.failure_reason:
            message += f": {reason}"
        raise _user_facing_error(message)


class AzureVideoAPIClient(BaseModel):
    creds: OpenAICreds
    base_url: str

    @property
    def _client(self) -> httpx.AsyncClient:
        return get_http_client()

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if key := self.creds.get("api_key"):
            headers["api_key"] = key
        if token := self.creds.get("azure_ad_token"):
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @property
    def _params(self) -> Dict[str, str]:
        return {"api-version": "preview"}

    @property
    def _client_options(self) -> dict:
        return {"headers": self._headers, "params": self._params}

    async def post_job(self, request: dict) -> VideoGenerationJob:
        url = f"{self.base_url}/jobs"
        resp = await self._client.post(
            url=url, json=request, **self._client_options
        )
        if not resp.is_success:
            raise _user_facing_error(
                "Video generation job creation failed", resp
            )

        return VideoGenerationJob.parse_obj(resp.json()).raise_on_failure()

    async def get_job_status(self, job_id: str) -> VideoGenerationJob:
        url = f"{self.base_url}/jobs/{job_id}"

        resp = await get_http_client().get(url=url, **self._client_options)

        if not resp.is_success:
            raise _user_facing_error(
                "Getting the status of a video generation job failed", resp
            )

        return VideoGenerationJob.parse_obj(resp.json()).raise_on_failure()

    async def get_video_content(self, generation_id: str) -> bytes:
        url = f"{self.base_url}/{generation_id}/content/video"

        resp = await get_http_client().get(url=url, **self._client_options)

        if not resp.is_success:
            raise _user_facing_error(
                "Fetching generated video content failed", resp
            )

        return resp.content


def _user_facing_error(
    message: str, response: httpx.Response | None = None
) -> InternalServerError:
    if response:
        logger.error(f"{message}: {response.status_code} {response.text}")
    else:
        logger.error(message)
    return InternalServerError(message=message, display_message=message)
