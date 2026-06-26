import base64
import contextlib
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import unquote, urljoin

import httpx
from aidial_client import (
    AsyncDial,
    DialException,
    InvalidDialURLError,
)
from aidial_client.types.metadata import FileMetadata
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger as log


@dataclass
class FileStorage:
    client: AsyncDial

    @classmethod
    def create(cls, dial_url: str, api_key: str) -> "FileStorage":
        from aidial_adapter_openai.app import get_dial_client_pool

        client = get_dial_client_pool().create_client(
            base_url=dial_url,
            api_key=api_key,
        )
        return cls(client=client)

    @property
    def dial_url(self) -> str:
        return self.client.base_url.removesuffix("/")

    def attachment_link_to_url(self, link: str) -> str:
        base_url = f"{self.client.base_url}v1/"
        return urljoin(base_url, link)

    def _url_to_attachment_link(self, url: str) -> str:
        return url.removeprefix(f"{self.client.base_url}v1/")

    def is_dial_url(self, link: str) -> bool:
        return self.client.is_dial_url(self.attachment_link_to_url(link))

    @staticmethod
    def _decode_link(link: str) -> str:
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = mimetypes.guess_extension(content_type) or ""
        stored_filename = f"{filename}{ext}"
        files_home = await self.client.my_files_home()
        upload_path = files_home / upload_dir / stored_filename

        metadata = await self.client.files.upload(
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

    async def download_file(self, link: str) -> bytes:
        url = self.attachment_link_to_url(link)

        try:
            try:
                result = await self.client.files.download(url=url)
                return await result.aget_content()
            except InvalidDialURLError:
                return await download_file(url)
        except DialException as e:
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {e.status_code})"
            ) from e
        except httpx.HTTPStatusError as e:
            raise InvalidRequestError(
                f"Failed to download file {link!r} (status code {e.response.status_code})"
            ) from e

    async def get_human_readable_name(self, link: str) -> str:
        with contextlib.suppress(Exception):
            link = self.client.files.get_display_name(link)

        return self._decode_link(link)


async def download_file(url: str) -> bytes:
    response = await get_http_client().get(url)
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

    return FileStorage.create(dial_url=DIAL_URL, api_key=api_key)
