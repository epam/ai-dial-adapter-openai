import mimetypes
from typing import Optional

from pydantic import BaseModel

from aidial_adapter_openai.utils.image import ImageDataURL
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import (
    FileStorage,
    download_file_as_base64,
)

# Officially supported image types by GPT-4 Vision, GPT-4o
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"]


def guess_attachment_type(attachment: dict) -> Optional[str]:
    type = attachment.get("type")

    if type is None or "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        url = attachment.get("url")
        if url is not None:
            url_type = mimetypes.guess_type(url)[0]
            if url_type is not None:
                return url_type

    return type


class ImageFail(BaseModel):
    class Config:
        frozen = True

    name: str
    message: str


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


async def download_image(
    file_storage: Optional[FileStorage], attachment: dict
) -> ImageDataURL | ImageFail:
    name = await get_attachment_name(file_storage, attachment)

    def fail(message: str) -> ImageFail:
        return ImageFail(name=name, message=message)

    try:
        type = guess_attachment_type(attachment)
        if type is None:
            return fail("can't derive media type of the attachment")
        elif type not in SUPPORTED_IMAGE_TYPES:
            return fail("the attachment is not one of the supported types")

        if "data" in attachment:
            return ImageDataURL(type=type, data=attachment["data"])

        if "url" in attachment:
            attachment_link: str = attachment["url"]

            image_url = ImageDataURL.from_data_url(attachment_link)
            if image_url is not None:
                if image_url.type not in SUPPORTED_IMAGE_TYPES:
                    return fail(
                        "the attachment is not one of the supported types"
                    )
                return image_url

            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return ImageDataURL(type=type, data=data)

        return fail("invalid attachment")

    except Exception as e:
        logger.error(f"Failed to download the image: {e}")
        return fail("failed to download the attachment")
