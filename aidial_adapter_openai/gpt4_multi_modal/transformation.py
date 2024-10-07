from typing import List, Optional, Tuple, TypeVar, cast

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.gpt4_multi_modal.attachment import (
    ImageFail,
    download_attachment_image,
    download_image_url,
)
from aidial_adapter_openai.utils.image import ImageMetadata
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import (
    MultiModalMessage,
    create_image_content_part,
    create_text_content_part,
)
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.storage import FileStorage
from aidial_adapter_openai.utils.text import decapitalize


class ImageProcessingFails(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    image_fails: List[ImageFail]


def create_image_metadata(
    arg: Resource | ImageFail,
) -> ImageMetadata | ImageFail:
    if isinstance(arg, Resource):
        return ImageMetadata.from_image_data_url(arg)
    return arg


async def download_attachment_images(
    file_storage: Optional[FileStorage], attachments: List[dict]
) -> List[ImageMetadata | ImageFail]:
    if attachments:
        logger.debug(f"original attachments: {attachments}")

    ret: List[ImageMetadata | ImageFail] = []

    for attachment in attachments:
        ret.append(
            create_image_metadata(
                await download_attachment_image(file_storage, attachment)
            )
        )

    return ret


async def download_content_images(
    file_storage: Optional[FileStorage], content: str | list
) -> List[ImageMetadata | ImageFail]:
    if isinstance(content, str):
        return []

    ret: List[ImageMetadata | ImageFail] = []

    for content_part in content:
        if image_url := content_part.get("image_url", {}).get("url"):
            ret.append(
                create_image_metadata(
                    await download_image_url(file_storage, image_url)
                )
            )

    return ret


_T = TypeVar("_T")


def _partition_errors(
    lst: List[_T | ImageFail],
) -> Tuple[List[_T], List[ImageFail]]:
    fails: List[ImageFail] = []
    images: List[_T] = []

    for elem in lst:
        if isinstance(elem, ImageFail):
            fails.append(elem)
        else:
            images.append(elem)

    return images, fails


async def transform_message(
    file_storage: Optional[FileStorage], message: dict
) -> MultiModalMessage | ImageProcessingFails:
    message = message.copy()

    content = message.get("content", "")
    custom_content = message.pop("custom_content", {})
    attachments = custom_content.get("attachments", [])

    attachment_metadatas, attachment_fails = _partition_errors(
        await download_attachment_images(file_storage, attachments)
    )

    content_metadatas, content_fails = _partition_errors(
        await download_content_images(file_storage, content)
    )

    if image_fails := [*attachment_fails, *content_fails]:
        logger.error(f"image processing errors: {image_fails}")
        return ImageProcessingFails(image_fails=image_fails)

    if not (image_metadatas := [*content_metadatas, *attachment_metadatas]):
        return MultiModalMessage(image_metadatas=[], raw_message=message)

    content_parts = (
        [create_text_content_part(content)]
        if isinstance(content, str)
        else content
    ) + [
        create_image_content_part(meta.image, meta.detail)
        for meta in attachment_metadatas
    ]

    return MultiModalMessage(
        image_metadatas=image_metadatas,
        raw_message={**message, "content": content_parts},
    )


async def transform_messages(
    file_storage: Optional[FileStorage], messages: List[dict]
) -> List[MultiModalMessage] | DialException:
    transformations = [
        await transform_message(file_storage, message) for message in messages
    ]

    errors = [
        error
        for error in transformations
        if isinstance(error, ImageProcessingFails)
    ]

    if errors:
        image_fails = sorted(
            set(
                image_fail
                for error in errors
                for image_fail in error.image_fails
            )
        )
        msg = "The following files failed to process:\n"
        msg += "\n".join(
            f"{idx}. {error.name}: {decapitalize(error.message)}"
            for idx, error in enumerate(image_fails, start=1)
        )
        return InvalidRequestError(message=msg, display_message=msg)

    transformations = cast(List[MultiModalMessage], transformations)
    return transformations
