import mimetypes
from typing import List, Optional, Set, Tuple, cast

from pydantic import BaseModel

from aidial_adapter_openai.gpt4_multi_modal.image_tokenizer import (
    tokenize_image,
)
from aidial_adapter_openai.gpt4_multi_modal.messages import (
    create_image_message,
    create_text_message,
)
from aidial_adapter_openai.utils.image_data_url import ImageDataURL
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


def get_attachment_name(
    file_storage: Optional[FileStorage], attachment: dict
) -> str:
    if "data" in attachment:
        return attachment.get("title") or "data attachment"

    if "url" in attachment:
        attachment_link = attachment["url"]
        if file_storage is not None:
            return file_storage.get_human_readable_name(attachment_link)
        return attachment_link

    return "invalid attachment"


async def download_image(
    file_storage: Optional[FileStorage], attachment: dict
) -> ImageDataURL | ImageFail:
    name = get_attachment_name(file_storage, attachment)

    def fail(message: str) -> ImageFail:
        return ImageFail(name=name, message=message)

    try:
        type = guess_attachment_type(attachment)
        if type is None:
            return fail("can't derive media type of the attachment")
        elif type not in SUPPORTED_IMAGE_TYPES:
            return fail(
                f"the attachment isn't one of the supported types: {type}"
            )

        if "data" in attachment:
            return ImageDataURL(type=type, data=attachment["data"])

        if "url" in attachment:
            attachment_link: str = attachment["url"]

            image_url = ImageDataURL.from_data_url(attachment_link)
            if image_url is not None:
                if image_url.type in SUPPORTED_IMAGE_TYPES:
                    return image_url
                else:
                    return fail(
                        f"the image attachment isn't one of the supported types: {image_url.type}"
                    )

            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return ImageDataURL(type=type, data=data)

        return fail("invalid attachment")

    except Exception as e:
        logger.error(f"Failed to download the image: {e}")
        return fail("failed to download the image")


async def transform_message(
    file_storage: Optional[FileStorage], message: dict
) -> Tuple[dict, int] | List[ImageFail]:
    content = message.get("content", "")
    custom_content = message.get("custom_content", {})
    attachments = custom_content.get("attachments", [])

    message = {k: v for k, v in message.items() if k != "custom_content"}

    if len(attachments) == 0:
        return message, 0

    logger.debug(f"original attachments: {attachments}")

    download_results: List[ImageDataURL | ImageFail] = [
        await download_image(file_storage, attachment)
        for attachment in attachments
    ]

    logger.debug(f"download results: {download_results}")

    errors: List[ImageFail] = [
        res for res in download_results if isinstance(res, ImageFail)
    ]

    if errors:
        logger.error(f"download errors: {errors}")
        return errors

    image_urls: List[ImageDataURL] = cast(List[ImageDataURL], download_results)

    image_tokens: List[int] = []
    image_messages: List[dict] = []

    for image_url in image_urls:
        tokens, detail = tokenize_image(image_url, "auto")
        image_tokens.append(tokens)
        image_messages.append(create_image_message(image_url, detail))

    total_image_tokens = sum(image_tokens)

    logger.debug(f"image tokens: {image_tokens}")

    sub_messages: List[dict] = [create_text_message(content)] + image_messages

    return {**message, "content": sub_messages}, total_image_tokens


async def transform_messages(
    file_storage: Optional[FileStorage], messages: List[dict]
) -> Tuple[List[dict], int] | str:
    image_tokens = 0
    new_messages: List[dict] = []
    errors: Set[ImageFail] = set()

    for message in messages:
        result = await transform_message(file_storage, message)
        if isinstance(result, list):
            errors.update(result)
        else:
            new_message, tokens = result
            new_messages.append(new_message)
            image_tokens += tokens

    if errors:
        msg = "The following file attachments failed to process:"
        for idx, error in enumerate(errors, start=1):
            msg += f"\n{idx}. {error.name}: {error.message}"
        return msg

    return new_messages, image_tokens
