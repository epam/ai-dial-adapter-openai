from urllib.parse import urlparse

from typing_extensions import override

from aidial_adapter_openai.dial_api.resource import ValidationError
from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.storage import Bucket, FileStorage


class MockFileStorage(FileStorage):
    def __init__(self):
        super().__init__(
            dial_url="http://dial-core",
            upload_dir="upload_dir",
            auth=Auth(name="api-key", value="dummy-api-key"),
        )

    @override
    async def _get_bucket(self, session) -> Bucket:
        return {
            "bucket": "APP_BUCKET",
            "appdata": "USER_BUCKET/appdata/test-application",
        }

    @override
    async def download_file(self, url: str) -> bytes:
        parsed_url = urlparse(url)
        if "not_found" in url:
            raise ValidationError("File not found")
        if not (parsed_url.scheme and parsed_url.netloc):
            raise ValidationError("Not a valid URL")
        return b"test-content"
