import base64
import hashlib
import mimetypes
import os
from typing import Mapping, Optional, TypedDict
from urllib.parse import unquote, urljoin

from pydantic import BaseModel

from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.env import get_env, get_env_bool
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger as log

CORE_API_VERSION = os.getenv("CORE_API_VERSION")


class FileMetadata(TypedDict):
    name: str
    parentPath: str
    bucket: str
    url: str


class Bucket(TypedDict):
    bucket: str
    appdata: str | None


class FileStorage(BaseModel):
    dial_url: str
    upload_dir: str
    auth: Auth

    bucket: Optional[Bucket] = None

    async def _get_bucket(self) -> Bucket:
        if self.bucket is None:
            response = await get_http_client().get(
                f"{self.dial_url}/v1/bucket",
                headers=self.auth.headers,
            )
            response.raise_for_status()
            self.bucket = response.json()
            log.debug(f"bucket: {self.bucket}")

        return self.bucket

    async def _get_user_bucket(self) -> str:
        bucket = await self._get_bucket()
        appdata = bucket.get("appdata")
        if appdata is None:
            raise ValueError(
                "Can't retrieve user bucket because appdata isn't available"
            )
        return appdata.split("/", 1)[0]

    async def upload(
        self, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        bucket = await self._get_bucket()

        appdata = bucket["appdata"]
        ext = mimetypes.guess_extension(content_type) or ""
        url = f"{self.dial_url}/v1/files/{appdata}/{self.upload_dir}/{filename}{ext}"

        response = await get_http_client().put(
            url=url,
            files={"file": (filename, content, content_type)},
            headers=self.auth.headers,
        )
        response.raise_for_status()
        meta = response.json()
        log.debug(f"Uploaded file: url={url}, metadata={meta}")
        return meta

    async def upload_file_as_base64(
        self, data: str, content_type: str
    ) -> FileMetadata:
        filename = _compute_hash_digest(data)
        content: bytes = base64.b64decode(data)
        return await self.upload(filename, content_type, content)

    def attachment_link_to_url(self, link: str) -> str:
        if CORE_API_VERSION == "0.6":
            base_url = f"{self.dial_url}/v1/files/"
        else:
            base_url = f"{self.dial_url}/v1/"

        return urljoin(base_url, link)

    def _url_to_attachment_link(self, url: str) -> str:
        if CORE_API_VERSION == "0.6":
            return url.removeprefix(f"{self.dial_url}/v1/files/")
        else:
            return url.removeprefix(f"{self.dial_url}/v1/")

    async def download_file(self, link: str) -> bytes:
        url = self.attachment_link_to_url(link)
        headers: Mapping[str, str] = {}
        if url.lower().startswith(self.dial_url.lower()):
            headers = self.auth.headers
        return await download_file(url, headers)

    async def get_human_readable_name(self, link: str) -> str:
        url = self.attachment_link_to_url(link)
        link = self._url_to_attachment_link(url)

        link = link.removeprefix("files/")

        if link.startswith("public/"):
            bucket = "public"
        else:
            bucket = await self._get_user_bucket()

        link = link.removeprefix(f"{bucket}/")
        decoded_link = unquote(link)
        return link if link == decoded_link else repr(decoded_link)


async def download_file(url: str, headers: Mapping[str, str] = {}) -> bytes:
    response = await get_http_client().get(url, headers=headers)
    response.raise_for_status()
    return response.read()


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
