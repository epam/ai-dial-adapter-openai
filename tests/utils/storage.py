import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse

from typing_extensions import override

from aidial_adapter_openai.dial_api.resource import ValidationError
from aidial_adapter_openai.dial_api.storage import (
    FileMetadata,
    FileStorage,
)
from aidial_adapter_openai.utils.env import get_env_bool


class DummyFileStorage(FileStorage):
    def __init__(self):
        super().__init__(
            client=FileStorage.create(
                dial_url="http://dial-core",
                api_key="dummy-api-key",
            ).client,
        )

    @override
    async def download_file(self, link: str) -> bytes:
        parsed_url = urlparse(link)
        if "not_found" in link:
            raise ValidationError("File not found")
        if not (parsed_url.scheme and parsed_url.netloc):
            raise ValidationError("Not a valid URL")
        return b"test-content"


class MockFileStorage(FileStorage):
    root_dir: Path
    files: list[Path]

    @classmethod
    def create_for_root(cls, root_dir: Path) -> "MockFileStorage":
        root_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            client=FileStorage.create(
                dial_url="http://test-dial-url",
                api_key="test-dial-api-key",
            ).client,
            root_dir=root_dir,
            files=[],
        )

    def _parse_filename(self, name: str) -> int:
        try:
            return int(name.split(".")[0])
        except Exception:
            return 0

    def _get_fresh_file_index(self) -> int:
        if not (files := os.listdir(self.root_dir)):
            return 1

        max_index = max(self._parse_filename(f) for f in files)
        return max_index + 1

    def _get_fresh_filename(self) -> str:
        return f"{self._get_fresh_file_index():0>3}"

    @staticmethod
    def _get_file_extension(content_type: str) -> str:
        return mimetypes.guess_extension(content_type) or ".bin"

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = self._get_file_extension(content_type)
        name = self._get_fresh_filename() + ext

        file = self.root_dir / name
        file.write_bytes(content)
        self.files.append(file)

        return FileMetadata(
            name=name,
            parent_path=os.path.dirname(name),
            bucket="mock-bucket",
            url=f"files/mock-bucket/{name}",
            node_type="ITEM",
            resource_type="FILE",
        )

    async def download_file(self, link: str) -> bytes:
        filename = link.removeprefix("files/mock-bucket/")
        return (self.root_dir / filename).read_bytes()

    async def get_human_readable_name(self, link: str) -> str:
        return link.removeprefix("files/mock-bucket/")

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if get_env_bool("INTEGRATION_TEST_CLEANUP_MOCK_STORAGE"):
            for file in self.files:
                file.unlink(missing_ok=True)

        if not os.listdir(self.root_dir):
            self.root_dir.rmdir()
