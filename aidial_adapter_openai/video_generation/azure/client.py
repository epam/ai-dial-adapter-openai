import json
from typing import Literal, NoReturn, Self

import httpx
from aidial_sdk.exceptions import InternalServerError, InvalidRequestError
from httpx._types import RequestFiles
from pydantic import BaseModel

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.video_generation.azure.types import (
    CreateVideoGenerationRequest,
    JobStatus,
    VideoGeneration,
)


class VideoGenerationJob(BaseModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/aae85aa3e7e4fda95ea2d3abac0ba1d8159db214/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L16123
    """

    id: str
    status: JobStatus
    generations: list[VideoGeneration] | None = None
    failure_reason: (
        str | Literal["input_moderation", "internal_error"] | None
    ) = None

    def raise_on_failure(self) -> Self:
        if self.status == JobStatus.FAILED:
            self.failed()
        return self

    def failed(self) -> NoReturn:
        assert self.status == JobStatus.FAILED

        message = "Video generation job failed"
        if reason := self.failure_reason:
            message += f": {reason}"

            if reason == "input_moderation":
                raise InvalidRequestError(
                    code="content_filter", message=message
                )

        raise InternalServerError(message=message)


class AzureVideoAPIClient(BaseModel):
    creds: OpenAICreds
    base_url: str

    @property
    def _client(self) -> httpx.AsyncClient:
        return get_http_client()

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if key := self.creds.get("api_key"):
            headers["api-key"] = key
        if token := self.creds.get("azure_ad_token"):
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @property
    def _params(self) -> dict[str, str]:
        return {"api-version": "preview"}

    @property
    def _client_options(self) -> dict:
        return {"headers": self._headers, "params": self._params}

    async def create_job(
        self, request: CreateVideoGenerationRequest, files: RequestFiles
    ) -> VideoGenerationJob:
        url = f"{self.base_url}/jobs"

        client_options = self._client_options
        if files:
            client_options["headers"].pop("Content-Type", None)

        request_body = request.model_dump(exclude_none=True)

        resp = await self._client.post(
            url=url,
            json=request_body if not files else None,
            data=request_body if files else None,
            files=files,
            **client_options,
        )
        resp.raise_for_status()

        resp_body = resp.json()
        logger.debug(f"job created: {json.dumps(resp_body)}")
        return VideoGenerationJob.model_validate(resp_body).raise_on_failure()

    async def get_job_status(self, job_id: str) -> VideoGenerationJob:
        url = f"{self.base_url}/jobs/{job_id}"

        resp = await get_http_client().get(url=url, **self._client_options)
        resp.raise_for_status()

        resp_body = resp.json()
        logger.debug(f"job status polled: {json.dumps(resp_body)}")
        return VideoGenerationJob.model_validate(resp_body).raise_on_failure()

    async def get_video_content(self, generation_id: str) -> bytes:
        url = f"{self.base_url}/{generation_id}/content/video"

        resp = await get_http_client().get(url=url, **self._client_options)
        resp.raise_for_status()
        return resp.content
