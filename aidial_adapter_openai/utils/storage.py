import base64
import hashlib
import io
from typing import Mapping, Optional, TypedDict

import aiohttp

from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.env import get_env, get_env_bool
from aidial_adapter_openai.utils.log_config import logger


class FileMetadata(TypedDict):
    name: str
    type: str
    path: str
    contentLength: int
    contentType: str


class FileStorage:
    dial_url: str
    bucket_url: str
    upload_url: str
    auth: Auth

    def __init__(self, dial_url: str, upload_dir: str, bucket: str, auth: Auth):
        self.dial_url = dial_url
        self.bucket_url = f"{self.dial_url}/v1/files/{bucket}"
        self.upload_url = f"{self.bucket_url}/{upload_dir}"
        self.auth = auth

    @classmethod
    async def create(cls, dial_url: str, base_dir: str, auth: Auth):
        bucket = await FileStorage._get_bucket(dial_url, auth)
        return cls(dial_url, base_dir, bucket, auth)

    def attachment_link_to_absolute_url(self, document_url_or_path: str) -> str:
        """
        Specification:
        - If URL starts with protocol "protocol://" then it's an URL.
        - If it starts with "/" then it's a relative path.
        - Otherwise, it's a corrupted link.
        - If it's a relative path, it needs to be "absolutized" first by concatenating "URL to core",
        "/v1/files" prefix for File API and "absolute" flag.

        Note: The dial does not follow RFC 1808 - Relative Uniform Resource Locators, so we cannot use urlparse here.
        """
        # Assume that "protocol://" means http or https protocol
        if document_url_or_path.startswith(
            "http://"
        ) or document_url_or_path.startswith("https://"):
            return document_url_or_path
        elif document_url_or_path.startswith("/"):
            return f"{self.bucket_url}{document_url_or_path}"
        else:
            raise ValueError(f"Corrupted link: {document_url_or_path}")

    @staticmethod
    async def _get_bucket(dial_url: str, auth: Auth) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{dial_url}/v1/bucket", headers=auth.headers
            ) as response:
                response.raise_for_status()
                body = await response.json()
                return body["bucket"]

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
            data = FileStorage._to_form_data(filename, content_type, content)
            async with session.put(
                f"{self.upload_url}/{filename}",
                data=data,
                headers=self.auth.headers,
            ) as response:
                response.raise_for_status()
                meta = await response.json()
                logger.debug(
                    f"Uploaded file: path={self.upload_url}, file={filename}, metadata={meta}"
                )
                return meta

    async def download(self, url: str) -> bytes:
        headers: Mapping[str, str] = {}
        if url.startswith(self.dial_url):
            headers = self.auth.headers

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.read()

    async def upload_file_as_base64(
        self, data: str, content_type: str
    ) -> FileMetadata:
        filename = _hash_digest(data)
        content: bytes = base64.b64decode(data)
        return await self.upload(filename, content_type, content)

    async def download_file_as_base64(self, url: str) -> str:
        bytes = await self.download(url)
        return base64.b64encode(bytes).decode("ascii")


def _hash_digest(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()


DIAL_USE_FILE_STORAGE = get_env_bool("DIAL_USE_FILE_STORAGE", False)

DIAL_URL: Optional[str] = None
if DIAL_USE_FILE_STORAGE:
    DIAL_URL = get_env(
        "DIAL_URL",
        "DIAL_URL environment variable must be initialized if DIAL_USE_FILE_STORAGE is true",
    )


async def create_file_storage(
    base_dir: str, headers: Mapping[str, str]
) -> Optional[FileStorage]:
    if not DIAL_USE_FILE_STORAGE:
        return None

    assert DIAL_URL is not None

    auth = Auth.from_headers("authorization", headers)

    if auth is None:
        logger.warning(
            "The request doesn't have required headers to use the DIAL file storage."
        )
        return None

    return await FileStorage.create(DIAL_URL, base_dir, auth)
