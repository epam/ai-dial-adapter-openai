import pytest

from aidial_adapter_openai.chat_completions.transformation import (
    Error,
    MessageTransformer,
)
from aidial_adapter_openai.dial_api.resource import (
    AttachmentResource,
    URLResource,
    parse_attachment,
)
from aidial_adapter_openai.utils.resource.base import Resource
from tests.utils.images import data_url, pic_1_1
from tests.utils.storage import DummyFileStorage


@pytest.fixture
def mock_message_transformer():
    return MessageTransformer(file_storage=DummyFileStorage())


@pytest.mark.parametrize(
    "url,expected_type",
    [
        ("image.jpg", "image/jpeg"),
        ("image.png", "image/png"),
        ("a/b/c/doc.txt", "text/plain"),
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
        (
            {"url": "http://dial-core/v1/image.png"},
            "http://dial-core/v1/image.png",
        ),
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
            DummyFileStorage()
        )
        == expected_name
    )


@pytest.mark.parametrize(
    "url, expected_result",
    [
        (data_url(pic_1_1), Resource.from_data_url(data_url(pic_1_1))),
        (
            "data:image/png;base65," + 1000 * "0",
            Error(
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
            Error(
                name="http://example.com/doc.pdf",
                message="The image is not one of the supported types",
            ),
        ),
        (
            "http://example.com/file.exotic_ext",
            Error(
                name="http://example.com/file.exotic_ext",
                message="Can't derive content type of the image",
            ),
        ),
    ],
)
async def test_download_image_url(
    mock_message_transformer: MessageTransformer,
    url: str,
    expected_result: Resource | Error,
):
    resource = URLResource(
        url=url,
        entity_name="image",
        supported_types=["image/png"],
    )
    result = await mock_message_transformer.try_download_resource(resource)
    if isinstance(expected_result, Resource):
        assert result == expected_result
    else:
        assert mock_message_transformer.errors == {expected_result}


@pytest.mark.parametrize(
    "attachment, expected_result",
    [
        ({"url": data_url(pic_1_1)}, Resource.from_data_url(data_url(pic_1_1))),
        (
            {"title": "attachment title", "data": "whatever"},
            Error(
                name="attachment title",
                message="Can't derive content type of the image",
            ),
        ),
        (
            {"type": "image/bmp", "url": data_url(pic_1_1)},
            Error(
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
            Error(
                name="data image",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"url": "data:image/png;base65,abcd"},
            Error(
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
            Error(
                name="http://example.com/doc.pdf",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"title": "PDF Document", "url": "http://example.com/doc.pdf"},
            Error(
                name="PDF Document",
                message="The image is not one of the supported types",
            ),
        ),
        (
            {"url": "http://example.com/file.exotic_ext"},
            Error(
                name="http://example.com/file.exotic_ext",
                message="Can't derive content type of the image",
            ),
        ),
    ],
)
async def test_download_attachment_image(
    mock_message_transformer: MessageTransformer,
    attachment: dict,
    expected_result: Resource | Error,
):
    resource = AttachmentResource(
        attachment=parse_attachment(attachment),
        entity_name="image",
        supported_types=["image/png"],
    )
    result = await mock_message_transformer.try_download_resource(resource)
    if isinstance(expected_result, Resource):
        assert result == expected_result
    else:
        assert mock_message_transformer.errors == {expected_result}
