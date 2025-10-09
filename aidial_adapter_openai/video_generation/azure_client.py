from typing import Dict

import httpx
from aidial_sdk.exceptions import InternalServerError
from pydantic import BaseModel

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger


def user_facing_error(
    message: str, response: httpx.Response | None = None
) -> InternalServerError:
    if response:
        logger.error(f"{message}: {response.status_code} {response.text}")
    else:
        logger.error(message)
    return InternalServerError(message=message, display_message=message)


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

    async def create_job(self, request: dict) -> dict:
        url = f"{self.base_url}/jobs"
        resp = await self._client.post(
            url=url, json=request, **self._client_options
        )

        if not resp.is_success:
            raise user_facing_error(
                "Video generation job creation failed", resp
            )

        return resp.json()

    async def get_job_status(self, job_id: str) -> dict:
        url = f"{self.base_url}/jobs/{job_id}"

        resp = await get_http_client().get(url=url, **self._client_options)

        if not resp.is_success:
            raise user_facing_error(
                "Getting the status of a video generation job failed", resp
            )

        return resp.json()

    async def download_video(self, generation_id: str) -> bytes:
        url = f"{self.base_url}/{generation_id}/content/video"

        resp = await get_http_client().get(url=url, **self._client_options)

        if not resp.is_success:
            raise user_facing_error(
                "Fetching generated video content failed", resp
            )

        return resp.content
