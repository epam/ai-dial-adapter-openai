import mimetypes
from typing import Callable, Optional

from aidial_adapter_openai.utils.image import ImageDataURL
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import (
    FileStorage,
    download_file_as_base64,
)

# Officially supported image types by GPT-4 Vision, GPT-4o
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"]


def guess_url_type(url: str) -> Optional[str]:
    return ImageDataURL.parse_content_type(url) or mimetypes.guess_type(url)[0]


def guess_attachment_type(attachment: dict) -> Optional[str]:
    type = attachment.get("type")

    if type is None or "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        if (url := attachment.get("url")) and (url_type := guess_url_type(url)):
            return url_type

    return type


class ImageFail(Exception):
    name: str
    message: str

    def __init__(self, name: str, message: str):
        self.name = name
        self.message = message


async def get_attachment_name(
    file_storage: Optional[FileStorage], attachment: dict
) -> str:
    if title := attachment.get("title"):
        return title

    if "data" in attachment:
        return attachment.get("title") or "data attachment"

    if "url" in attachment:
        link = attachment["url"]
        if file_storage is not None:
            return await file_storage.get_human_readable_name(link)
        return link

    return "invalid attachment"


def _validate_image_type(
    content_type: Optional[str], fail: Callable[[str], ImageFail]
) -> str:
    if content_type is None:
        raise fail("can't derive media type of the image")

    if content_type not in SUPPORTED_IMAGE_TYPES:
        raise fail("the image is not one of the supported types")

    return content_type


async def download_image_url(
    file_storage: Optional[FileStorage],
    image_link: str,
    *,
    override_fail: Optional[Callable[[str], ImageFail]] = None,
    override_type: Optional[str] = None,
) -> ImageDataURL | ImageFail:
    """
    The image link is either a URL of the image (public of DIAL) or the base64 encoded image data.
    """

    def fail(message: str) -> ImageFail:
        return ImageFail(name="image_url", message=message)

    try:
        type = _validate_image_type(
            override_type or guess_url_type(image_link),
            override_fail or fail,
        )

        image_url = ImageDataURL.from_data_url(image_link)
        if image_url is not None:
            return image_url

        if file_storage is not None:
            url = file_storage.attachment_link_to_url(image_link)
            data = await file_storage.download_file_as_base64(url)
        else:
            data = await download_file_as_base64(image_link)

        return ImageDataURL(type=type, data=data)

    except ImageFail as e:
        return e

    except Exception as e:
        logger.error(f"Failed to download the image: {e}")
        return fail("failed to download the image")


async def download_attachment_image(
    file_storage: Optional[FileStorage], attachment: dict
) -> ImageDataURL | ImageFail:
    name = await get_attachment_name(file_storage, attachment)

    def fail(message: str) -> ImageFail:
        return ImageFail(name=name, message=message)

    try:
        type = _validate_image_type(guess_attachment_type(attachment), fail)

        if "data" in attachment:
            return ImageDataURL(type=type, data=attachment["data"])

        if "url" in attachment:
            return await download_image_url(
                file_storage,
                attachment["url"],
                override_fail=fail,
                override_type=type,
            )

        raise fail("invalid attachment")

    except ImageFail as e:
        return e

    except Exception as e:
        logger.error(f"Failed to download the attachment: {e}")
        return fail("failed to download the attachment")
