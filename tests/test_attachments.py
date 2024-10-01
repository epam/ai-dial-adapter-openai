import pytest
from typing_extensions import override

from aidial_adapter_openai.gpt4_multi_modal.attachment import (
    get_attachment_name,
    guess_attachment_type,
    guess_url_type,
)
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


@pytest.mark.parametrize(
    "url,expected_type",
    [
        ("image.jpg", "image/jpeg"),
        ("image.png", "image/png"),
        ("a/b/c/doc.text", "text/plain"),
        ("dir1/dir2/", None),
        ("no_ext", None),
        ("binary.a", "application/octet-stream"),
        ("unknown.x", None),
        ("data:image/png;base64,abcd", "image/png"),
        ("data:whatever;base64,abcd", "whatever"),
        ("data:what/ever;base64,abcd", "what/ever"),
        (
            "data:image/png;base65,abcd",
            # mimetypes.guess_type analyses only the "data:{type}" prefix of the string
            "image/png",
        ),
    ],
)
def test_guess_url_type(url, expected_type):
    assert guess_url_type(url) == expected_type


@pytest.mark.parametrize(
    "attachment, expected_type",
    [
        ({"type": "image/png", "url": "whatever"}, "image/png"),
        ({"type": None, "url": "x/y/z.txt"}, "text/plain"),
        (
            {"type": "application/octet-stream", "url": "x/y/z.gif"},
            "image/gif",
        ),
        (
            {"type": "application/octet-stream", "data": "abcd"},
            "application/octet-stream",
        ),
        (
            {"type": None, "data": "abcd"},
            None,
        ),
    ],
)
def test_guess_attachment_type(attachment, expected_type):
    assert guess_attachment_type(attachment) == expected_type


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attachment, expected_name",
    [
        ({"url": "what"}, "what"),
        ({"url": "relative/url.gif"}, "relative/url.gif"),
        ({"data": "abcd"}, "data attachment"),
        ({"url": "http://dial-core/image.png"}, "http://dial-core/image.png"),
        ({"url": "http://dial-core/v1/image.png"}, "image.png"),
        (
            {"url": "http://dial-core/v1/USER_BUCKET/dir1/dir2/image.png"},
            "dir1/dir2/image.png",
        ),
        (
            {"url": "http://dial-core/v1/public/dir1/dir2/image.png"},
            "dir1/dir2/image.png",
        ),
        (
            {"url": "http://dial-core/v1/public/dir1/dir2/hello%20world.png"},
            "'dir1/dir2/hello world.png'",
        ),
    ],
)
async def test_get_attachment_name(attachment, expected_name):
    assert (
        await get_attachment_name(MockFileStorage(), attachment)
        == expected_name
    )
