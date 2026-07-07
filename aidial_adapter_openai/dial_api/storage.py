import base64
import contextlib
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote

import httpx
from aidial_client import AsyncDial, DialException
from aidial_client._exception import NotDialURLError
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

    def is_dial_url(self, link: str) -> bool:
        try:
            self.client.files.get_storage_resource(link)
            return True
        except NotDialURLError:
            return False

    @staticmethod
    def _decode_link(link: str) -> str:
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)

    async def _upload_base_dir(self) -> PurePosixPath:
        appdata = await self.client.my_appdata_home()
        if appdata is None:
            raise ValueError("Unable to retrieve user appdata directory.")
        return "files" / appdata

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = mimetypes.guess_extension(content_type) or ""
        stored_filename = f"{filename}{ext}"
        base_dir = await self._upload_base_dir()
        upload_path = base_dir / upload_dir / stored_filename

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
        try:
            try:
                result = await self.client.files.download(url=link)
                return await result.aget_content()
            except NotDialURLError:
                return await download_file(link)
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
