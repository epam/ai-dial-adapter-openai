import os
import shutil
from pathlib import Path
from typing import Dict

from pydantic import SecretStr

from aidial_adapter_openai.dial_api.storage import FileMetadata, FileStorage


class MockFileStorage(FileStorage):
    base_dir: Path
    files_cache: Dict[str, bytes]

    @classmethod
    def create(cls, base_dir: Path) -> "MockFileStorage":
        return cls(
            dial_url="http://mock",
            api_key=SecretStr("mock"),
            base_dir=base_dir,
            files_cache={},
        )

    async def upload(
        self, upload_dir: str, filename: str, content_type: str, content: bytes
    ) -> FileMetadata:
        ext = ".png" if content_type == "image/png" else ".jpeg"
        full_path = self.base_dir / upload_dir / (filename + ext)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)

        self.files_cache[filename] = content

        return FileMetadata(
            name=filename,
            parentPath=os.path.dirname(filename),
            bucket="mock-bucket",
            url=f"files/mock-bucket/{filename}",
        )

    async def download_file(self, link: str) -> bytes:
        filename = link.removeprefix("files/mock-bucket/")
        return self.files_cache[filename]

    async def get_human_readable_name(self, link: str) -> str:
        return link.removeprefix("files/mock-bucket/")

    def cleanup(self):
        shutil.rmtree(self.base_dir)
