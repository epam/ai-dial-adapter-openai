import base64
from urllib.parse import urlparse

import pytest
from typing_extensions import override

from aidial_adapter_openai.gpt4_multi_modal.attachment import (
    ImageFail,
    download_attachment_image,
    download_image_url,
    get_attachment_name,
    guess_attachment_type,
    guess_url_type,
)
from aidial_adapter_openai.utils.auth import Auth
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.storage import Bucket, FileStorage
from tests.utils.images import data_url, pic_1_1


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
    async def download_file_as_base64(self, url: str) -> str:
        parsed_url = urlparse(url)
        if not (parsed_url.scheme and parsed_url.netloc):
            raise RuntimeError("Not a valid URL")
        return base64.b64encode("test-content".encode()).decode("ascii")


@pytest.mark.parametrize(
    "url,expected_type",
    [
        ("image.jpg", "image/jpeg"),
        ("image.png", "image/png"),
        ("a/b/c/doc.text", "text/plain"),
        ("dir1/dir2/", None),
        ("no_ext", None),
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
        ({"title": "attachment title", "url": "whatever"}, "attachment title"),
        ({"url": "what"}, "what"),
        ({"url": "relative/url.gif"}, "relative/url.gif"),
        ({"data": "abcd"}, "data attachment"),
        ({"url": "http://dial-core/image.png"}, "http://dial-core/image.png"),
        ({"url": "http://dial-core/v1/image.png"}, "image.png"),
        (
            {
                "url": "http://dial-core/v1/files/USER_BUCKET/dir1/dir2/image.png"
            },
            "dir1/dir2/image.png",
        ),
        (
            {"url": "http://dial-core/v1/files/public/dir1/dir2/image.png"},
            "dir1/dir2/image.png",
        ),
        (
            {
                "url": "http://dial-core/v1/files/public/dir1/dir2/hello%20world.png"
            },
            "'dir1/dir2/hello world.png'",
        ),
    ],
)
async def test_get_attachment_name(attachment, expected_name):
    assert (
        await get_attachment_name(MockFileStorage(), attachment)
        == expected_name
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, expected_result",
    [
        (data_url(pic_1_1), Resource.from_data_url(data_url(pic_1_1))),
        (
            "data:image/png;base65,abcd",
            ImageFail(name="image_url", message="failed to download the image"),
        ),
        (
            "http://example.com/image.png",
            Resource(type="image/png", data=b"test-content"),
        ),
        (
            "http://example.com/doc.pdf",
            ImageFail(
                name="image_url",
                message="the image is not one of the supported types",
            ),
        ),
        (
            "http://example.com/file.exotic_ext",
            ImageFail(
                name="image_url",
                message="can't derive media type of the image",
            ),
        ),
    ],
)
async def test_download_image_url(url, expected_result):
    assert await download_image_url(MockFileStorage(), url) == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, expected_result",
    [
        ({"url": data_url(pic_1_1)}, Resource.from_data_url(data_url(pic_1_1))),
        (
            {"title": "attachment title"},
            ImageFail(
                name="attachment title",
                message="can't derive media type of the image",
            ),
        ),
        (
            {"url": data_url(pic_1_1), "type": "image/bmp"},
            ImageFail(
                name="data URL",
                message="the image is not one of the supported types",
            ),
        ),
        (
            {"data": pic_1_1, "type": "image/png"},
            Resource.from_data_url(data_url(pic_1_1)),
        ),
        (
            {"data": pic_1_1, "type": "image/bmp"},
            ImageFail(
                name="data attachment",
                message="the image is not one of the supported types",
            ),
        ),
        (
            {"url": "data:image/png;base65,abcd"},
            ImageFail(name="image_url", message="failed to download the image"),
        ),
        (
            {"url": "data:image/png;base65,abcd"},
            ImageFail(name="image_url", message="failed to download the image"),
        ),
        (
            {"url": "http://example.com/image.png"},
            Resource(type="image/png", data=b"test-content"),
        ),
        (
            {"url": "http://example.com/doc.pdf"},
            ImageFail(
                name="http://example.com/doc.pdf",
                message="the image is not one of the supported types",
            ),
        ),
        (
            {"title": "PDF Document", "url": "http://example.com/doc.pdf"},
            ImageFail(
                name="PDF Document",
                message="the image is not one of the supported types",
            ),
        ),
        (
            {"url": "http://example.com/file.exotic_ext"},
            ImageFail(
                name="http://example.com/file.exotic_ext",
                message="can't derive media type of the image",
            ),
        ),
    ],
)
async def test_download_attachment_image(url, expected_result):
    assert (
        await download_attachment_image(MockFileStorage(), url)
        == expected_result
    )
