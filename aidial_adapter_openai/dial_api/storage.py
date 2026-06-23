import base64
import hashlib
import mimetypes
import os
from collections.abc import Mapping
from pathlib import PurePosixPath
from urllib.parse import unquote, urljoin

import httpx
from aidial_client import AsyncDial, DialException
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
    _my_files_home: PurePosixPath | None = PrivateAttr(default=None)

    @property
    def headers(self) -> Mapping[str, str]:
        return {"api-key": self.api_key.get_secret_value()}

    def _get_dial_client(self) -> AsyncDial:
        if self._dial_client is not None:
            return self._dial_client

        self._dial_client = AsyncDial(
            base_url=self.dial_url,
            api_key=self.api_key.get_secret_value(),
        )
        return self._dial_client

    async def _get_my_files_home(self) -> PurePosixPath:
        if self._my_files_home is not None:
            return self._my_files_home

        self._my_files_home = await self._get_dial_client().my_files_home()
        return self._my_files_home

    @staticmethod
    def _to_file_metadata(meta: SDKFileMetadata) -> FileMetadata:
        return {
            "name": meta.name or "",
            "parentPath": meta.parent_path or "",
            "bucket": meta.bucket or "",
            "url": meta.url or "",
        }

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = mimetypes.guess_extension(content_type) or ""
        stored_filename = f"{filename}{ext}"
        files_home = await self._get_my_files_home()
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

    def _is_dial_file_url(self, url: str) -> bool:
        return self._url_to_attachment_link(url).startswith("files/")

    @staticmethod
    def _to_human_readable_name(name: str) -> str:
        decoded_name = unquote(name)
        return name if name == decoded_name else repr(decoded_name)

    async def download_file(self, link: str) -> bytes:
        url = self.attachment_link_to_url(link)
        headers: Mapping[str, str] = {}
        if self.is_dial_url(link):
            headers = self.headers

        try:
            if self._is_dial_file_url(url):
                result = await self._get_dial_client().files.download(url=url)
                return await result.aget_content()

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
        if self._is_dial_file_url(url):
            try:
                name = self._get_dial_client().files.get_display_name(url)
                return self._to_human_readable_name(name)
            except DialException:
                pass

        name = self._url_to_attachment_link(url)
        return self._to_human_readable_name(name)


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
