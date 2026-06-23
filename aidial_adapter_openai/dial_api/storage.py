import base64
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from urllib.parse import unquote, urljoin

import httpx
from aidial_client import AsyncDial, DialException, InvalidDialURLError
from aidial_client._constants import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT
from aidial_client._http_client._async import AsyncHTTPClient
from aidial_client.types.metadata import FileMetadata as SDKFileMetadata
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel, PrivateAttr, SecretStr
from typing_extensions import TypedDict

from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger as log


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str


class FileStorage(BaseModel):
    dial_url: str
    api_key: SecretStr

    _dial_client: AsyncDial | None = PrivateAttr(default=None)

    @property
    def headers(self) -> Mapping[str, str]:
        return {"api-key": self.api_key.get_secret_value()}

    def _get_dial_client(self) -> AsyncDial:
        if self._dial_client is not None:
            return self._dial_client

        sdk_http_client = AsyncHTTPClient(
            base_url=self.dial_url,
            api_key=self.api_key.get_secret_value(),
            bearer_token=None,
            max_retries=DEFAULT_MAX_RETRIES,
            timeout=DEFAULT_TIMEOUT,
            internal_http_client=get_http_client(),
        )

        self._dial_client = AsyncDial(
            base_url=self.dial_url,
            api_key=self.api_key.get_secret_value(),
            http_client=sdk_http_client,
        )
        return self._dial_client

    @staticmethod
    def _to_file_metadata(meta: SDKFileMetadata) -> FileMetadata:
        metadata = meta.model_dump()
        return {
            "name": metadata["name"],
            "parentPath": metadata["parent_path"],
            "bucket": metadata["bucket"],
            "url": metadata["url"],
        }

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = mimetypes.guess_extension(content_type) or ""
        stored_filename = f"{filename}{ext}"
        files_home = await self._get_dial_client().my_files_home()
        upload_path = files_home / upload_dir / stored_filename

        metadata = await self._get_dial_client().files.upload(
            url=upload_path,
            file=(stored_filename, content, content_type),
        )
        metadata_ = self._to_file_metadata(metadata)
        log.debug(f"Uploaded file: url={upload_path}, metadata={metadata_}")
        return metadata_

    async def upload_file(
        self, upload_dir: str, data: str | bytes, content_type: str
    ) -> FileMetadata:
        filename = _compute_hash_digest(data)
        if isinstance(data, str):
            content: bytes = base64.b64decode(data)
        else:
            content = data
        return await self.upload(upload_dir, filename, content_type, content)

    def attachment_link_to_url(self, link: str) -> str:
        base_url = f"{self.dial_url}/v1/"
        return urljoin(base_url, link)

    def _url_to_attachment_link(self, url: str) -> str:
        return url.removeprefix(f"{self.dial_url}/v1/")

    def is_dial_url(self, link: str) -> bool:
        url = self.attachment_link_to_url(link)
        return url.lower().startswith(self.dial_url.lower())

    async def download_file(self, link: str) -> bytes:
        url = self.attachment_link_to_url(link)
        headers = self.headers if self.is_dial_url(link) else None

        try:
            if self.is_dial_url(link):
                try:
                    result = await self._get_dial_client().files.download(
                        url=url
                    )
                    return await result.aget_content()
                except InvalidDialURLError:
                    pass

            return await download_file(url, headers)
        except DialException as e:
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {e.status_code})"
            ) from e
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {code})"
            ) from e

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        if self.is_dial_url(link):
            try:
                link = self._get_dial_client().files.get_display_name(url)
            except InvalidDialURLError:
                link = self._url_to_attachment_link(url)
        else:
            link = self._url_to_attachment_link(url)

        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)


async def download_file(
    url: str, headers: Mapping[str, str] | None = None
) -> bytes:
    response = await get_http_client().get(url, headers=headers)
    response.raise_for_status()
    return response.read()


def _compute_hash_digest(file_content: str | bytes) -> str:
    if isinstance(file_content, str):
        file_content = file_content.encode()
    return hashlib.sha256(file_content).hexdigest()


DIAL_URL = os.getenv("DIAL_URL")


def create_file_storage(headers: Mapping[str, str]) -> FileStorage | None:
    if DIAL_URL is None:
        return None

    if (api_key := headers.get("api-key")) is None:
        log.debug(
            "The request doesn't have required headers to use the DIAL file storage. "
            "Fallback to base64 encoding of images."
        )
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=SecretStr(api_key))
