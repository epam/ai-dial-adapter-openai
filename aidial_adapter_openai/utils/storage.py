import base64
import hashlib
import io
import mimetypes
import os
from typing import Mapping, Optional, TypedDict
from urllib.parse import unquote, urljoin

import aiohttp

from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.env import get_env, get_env_bool
from aidial_adapter_openai.utils.log_config import logger as log

core_api_version = os.environ.get("CORE_API_VERSION", None)


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str


class Bucket(TypedDict):
    bucket: str
    appdata: str | None


class FileStorage:
    dial_url: str
    upload_dir: str
    auth: Auth

    bucket: Optional[Bucket]

    def __init__(self, dial_url: str, upload_dir: str, auth: Auth):
        self.dial_url = dial_url
        self.upload_dir = upload_dir
        self.auth = auth
        self.bucket = None

    async def _get_bucket(self, session: aiohttp.ClientSession) -> Bucket:
        if self.bucket is None:
            async with session.get(
                f"{self.dial_url}/v1/bucket",
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                self.bucket = await response.json()
                log.debug(f"bucket: {self.bucket}")

        return self.bucket

    async def _get_user_bucket(self, session: aiohttp.ClientSession) -> str:
        bucket = await self._get_bucket(session)
        appdata = bucket.get("appdata")
        if appdata is None:
            raise ValueError(
                "Can't retrieve user bucket because appdata isn't available"
            )
        return appdata.split("/", 1)[0]

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

            appdata = bucket["appdata"]
            ext = mimetypes.guess_extension(content_type) or ""
            url = f"{self.dial_url}/v1/files/{appdata}/{self.upload_dir}/{filename}{ext}"

            data = FileStorage._to_form_data(filename, content_type, content)

            async with session.put(
                url=url,
                data=data,
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                log.debug(f"Uploaded file: url={url}, metadata={meta}")
                return meta

    async def upload_file_as_base64(
        self, data: str, content_type: str
    ) -> FileMetadata:
        filename = _compute_hash_digest(data)
        content: bytes = base64.b64decode(data)
        return await self.upload(filename, content_type, content)

    def attachment_link_to_url(self, link: str) -> str:
        if core_api_version == "0.6":
            base_url = f"{self.dial_url}/v1/files/"
        else:
            base_url = f"{self.dial_url}/v1/"

        return urljoin(base_url, link)

    def _url_to_attachment_link(self, url: str) -> str:
        if core_api_version == "0.6":
            return url.removeprefix(f"{self.dial_url}/v1/files/")
        else:
            return url.removeprefix(f"{self.dial_url}/v1/")

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        link = self._url_to_attachment_link(url)

        if link.startswith("public/"):
            bucket = "public"
        else:
            async with aiohttp.ClientSession() as session:
                bucket = await self._get_user_bucket(session)

        link = link.removeprefix(f"{bucket}/")
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(link)

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
        log.debug(
            "The request doesn't have required headers to use the DIAL file storage. "
            "Fallback to base64 encoding of images."
        )
        return None

    return FileStorage(dial_url=DIAL_URL, upload_dir=base_dir, auth=auth)
