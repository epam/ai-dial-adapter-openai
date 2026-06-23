import base64
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from urllib.parse import unquote, urljoin

import httpx
from aidial_client import (
    AsyncDial,
    AsyncDialClientPool,
    DialException,
    InvalidDialURLError,
)
from aidial_client.types.metadata import FileMetadata
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel, PrivateAttr, SecretStr

from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger as log

_DIAL_CLIENT_POOL = AsyncDialClientPool()


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

        self._dial_client = _DIAL_CLIENT_POOL.create_client(
            base_url=self.dial_url,
            api_key=self.api_key.get_secret_value(),
        )
        return self._dial_client

    @staticmethod
    def _decode_link(link: str) -> str:
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        dial_client = self._get_dial_client()
        ext = mimetypes.guess_extension(content_type) or ""
        stored_filename = f"{filename}{ext}"
        files_home = await dial_client.my_files_home()
        upload_path = files_home / upload_dir / stored_filename

        metadata = await dial_client.files.upload(
            url=upload_path,
            file=(stored_filename, content, content_type),
        )
        log.debug(f"Uploaded file: url={upload_path}, metadata={metadata}")
        return metadata

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
        is_dial_link = self.is_dial_url(link)

        try:
            if is_dial_link:
                try:
                    result = await self._get_dial_client().files.download(
                        url=url
                    )
                    return await result.aget_content()
                except InvalidDialURLError:
                    pass

            headers = self.headers if is_dial_link else None
            return await download_file(url, headers)
        except DialException as e:
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {e.status_code})"
            )
        except httpx.HTTPStatusError as e:
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {e.response.status_code})"
            )

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        attachment_link = self._url_to_attachment_link(url)

        if not self.is_dial_url(link):
            return self._decode_link(attachment_link)

        try:
            link = self._get_dial_client().files.get_display_name(url)
        except InvalidDialURLError:
            link = attachment_link

        return self._decode_link(link)


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
