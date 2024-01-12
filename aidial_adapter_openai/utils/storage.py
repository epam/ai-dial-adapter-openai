import base64
import hashlib
import io
import mimetypes
from typing import Mapping, Optional, TypedDict
from urllib.parse import urljoin

import aiohttp

from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.env import get_env, get_env_bool
from aidial_adapter_openai.utils.log_config import logger


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str
    type: str

    contentLength: int
    contentType: str


class FileStorage:
    dial_url: str
    upload_dir: str
    auth: Auth
    bucket: Optional[str]

    def __init__(self, dial_url: str, upload_dir: str, auth: Auth):
        self.dial_url = dial_url
        self.upload_dir = upload_dir
        self.auth = auth
        self.bucket = None

    async def _get_bucket(self, session: aiohttp.ClientSession) -> str:
        if self.bucket is None:
            async with session.get(
                f"{self.dial_url}/v1/bucket",
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                body = await response.json()
                self.bucket = body["bucket"]

        return self.bucket

    @staticmethod
    def _to_form_data(
        filename: str, content_type: str, content: bytes
    ) -> aiohttp.FormData:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            io.BytesIO(content),
            filename=filename,
            content_type=content_type,
        )
        return data

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            bucket = await self._get_bucket(session)
            data = FileStorage._to_form_data(filename, content_type, content)
            ext = mimetypes.guess_extension(content_type) or ""
            url = f"{self.dial_url}/v1/files/{bucket}/{self.upload_dir}/{filename}{ext}"

            async with session.put(
                url=url,
                data=data,
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                logger.debug(f"Uploaded file: url={url}, metadata={meta}")
                return meta

    async def upload_file_as_base64(
        self, data: str, content_type: str
    ) -> FileMetadata:
        filename = _compute_hash_digest(data)
        content: bytes = base64.b64decode(data)
        return await self.upload(filename, content_type, content)

    def attachment_link_to_url(self, link: str) -> str:
        base_url = f"{self.dial_url}/v1/files/"
        return urljoin(base_url, link)

    async def download_file_as_base64(self, url: str) -> str:
        headers: Mapping[str, str] = {}
        if url.startswith(self.dial_url):
            headers = self.auth.headers

        return await download_file_as_base64(url, headers)


async def _download_file(url: str, headers: Mapping[str, str] = {}) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return await response.read()


async def download_file_as_base64(
    url: str, headers: Mapping[str, str] = {}
) -> str:
    bytes = await _download_file(url, headers)
    return base64.b64encode(bytes).decode("ascii")


def _compute_hash_digest(file_content: str) -> str:
    return hashlib.sha256(file_content.encode()).hexdigest()


DIAL_USE_FILE_STORAGE = get_env_bool("DIAL_USE_FILE_STORAGE", False)

DIAL_URL: Optional[str] = None
if DIAL_USE_FILE_STORAGE:
    DIAL_URL = get_env(
        "DIAL_URL", "DIAL_URL must be set to use the DIAL file storage"
    )


def create_file_storage(
    base_dir: str, headers: Mapping[str, str]
) -> Optional[FileStorage]:
    if not DIAL_USE_FILE_STORAGE or DIAL_URL is None:
        return None

    auth = Auth.from_headers("api-key", headers)
    if auth is None:
        logger.debug(
            "The request doesn't have required headers to use the DIAL file storage. "
            "Fallback to base64 encoding of images."
        )
        return None

    return FileStorage(dial_url=DIAL_URL, upload_dir=base_dir, auth=auth)
