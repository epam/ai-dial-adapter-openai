import pytest

from aidial_adapter_openai.dial_api.resource import (
    AttachmentResource,
    URLResource,
    parse_attachment,
)
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    ResourceProcessor,
    TransformationError,
)
from aidial_adapter_openai.utils.resource import Resource
from tests.utils.images import data_url, pic_1_1
from tests.utils.storage import MockFileStorage


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
async def test_guess_url_type(url, expected_type):
    assert await URLResource(url=url).guess_content_type() == expected_type


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
async def test_guess_attachment_type(attachment, expected_type):
    assert (
        await AttachmentResource(attachment=attachment).guess_content_type()
        == expected_type
    )


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
        await AttachmentResource(attachment=attachment).get_resource_name(
            MockFileStorage()
        )
        == expected_name
    )


@pytest.mark.parametrize(
    "url, expected_result",
    [
        (data_url(pic_1_1), Resource.from_data_url(data_url(pic_1_1))),
        (
            "data:image/png;base65," + 1000 * "0",
            TransformationError(
                name="data:image/png;base65,0000000000000000000000000000...",
                message="Not a valid URL",
            ),
        ),
        (
            "http://example.com/image.png",
            Resource(type="image/png", data=b"test-content"),
        ),
        (
            "http://example.com/doc.pdf",
            TransformationError(
                name="http://example.com/doc.pdf",
                message="The image is not one of the supported types",
            ),
        ),
        (
            "http://example.com/file.exotic_ext",
            TransformationError(
                name="http://example.com/file.exotic_ext",
                message="Can't derive content type of the image",
            ),
        ),
    ],
)
async def test_download_image_url(url, expected_result):
    resource = URLResource(
        url=url,
        entity_name="image",
        supported_types=["image/png"],
    )
    processor = ResourceProcessor(file_storage=MockFileStorage())
    assert await processor.try_download_resource(resource) == expected_result


@pytest.mark.parametrize(
    "attachment, expected_result",
    [
        ({"url": data_url(pic_1_1)}, Resource.from_data_url(data_url(pic_1_1))),
        (
            {"title": "attachment title"},
            TransformationError(
                name="attachment title",
                message="Can't derive content type of the image",
            ),
        ),
        (
            {"type": "image/bmp", "url": data_url(pic_1_1)},
            TransformationError(
                name="data URL (image/bmp)",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"type": "image/png", "data": pic_1_1.data_base64},
            Resource.from_data_url(data_url(pic_1_1)),
        ),
        (
            {"type": "image/bmp", "data": pic_1_1.data_base64},
            TransformationError(
                name="data image",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"url": "data:image/png;base65,abcd"},
            TransformationError(
                name="data:image/png;base65,abcd",
                message="Not a valid URL",
            ),
        ),
        (
            {"url": "http://example.com/image.png"},
            Resource(type="image/png", data=b"test-content"),
        ),
        (
            {"url": "http://example.com/doc.pdf"},
            TransformationError(
                name="http://example.com/doc.pdf",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"title": "PDF Document", "url": "http://example.com/doc.pdf"},
            TransformationError(
                name="PDF Document",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"url": "http://example.com/file.exotic_ext"},
            TransformationError(
                name="http://example.com/file.exotic_ext",
                message="Can't derive content type of the image",
            ),
        ),
    ],
)
async def test_download_attachment_image(attachment: dict, expected_result):
    resource = AttachmentResource(
        attachment=parse_attachment(attachment),
        entity_name="image",
        supported_types=["image/png"],
    )
    processor = ResourceProcessor(file_storage=MockFileStorage())
    assert await processor.try_download_resource(resource) == expected_result
