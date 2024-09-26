from typing import List, Optional, cast

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.gpt4_multi_modal.attachment import (
    ImageFail,
    download_image,
)
from aidial_adapter_openai.utils.image import ImageDataURL, ImageMetadata
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import (
    MultiModalMessage,
    create_image_content_part,
    create_text_content_part,
)
from aidial_adapter_openai.utils.storage import FileStorage


class TransformationError(BaseModel):
    image_fails: List[ImageFail]


async def transform_message(
    file_storage: Optional[FileStorage],
    message: dict,
) -> MultiModalMessage | TransformationError:
    content = message.get("content", "")
    custom_content = message.get("custom_content", {})
    attachments = custom_content.get("attachments", [])
    logger.debug(f"original attachments: {attachments}")

    message = {k: v for k, v in message.items() if k != "custom_content"}

    download_results: List[ImageDataURL | ImageFail] = [
        await download_image(file_storage, attachment)
        for attachment in attachments
    ]

    logger.debug(f"download results: {download_results}")

    fails: List[ImageFail] = [
        res for res in download_results if isinstance(res, ImageFail)
    ]

    if fails:
        logger.error(f"download errors: {fails}")
        return TransformationError(image_fails=fails)

    image_urls: List[ImageDataURL] = cast(List[ImageDataURL], download_results)
    image_metadatas = [
        ImageMetadata.from_image_data_url(image_url) for image_url in image_urls
    ]
    if image_metadatas:
        content = [create_text_content_part(content)] + [
            create_image_content_part(
                image=image_metadata.image,
                detail=image_metadata.detail,
            )
            for image_metadata in image_metadatas
        ]

    return MultiModalMessage(
        image_metadatas=image_metadatas,
        raw_message={
            **message,
            "content": content,
        },
    )


async def transform_messages(
    file_storage: Optional[FileStorage],
    messages: List[dict],
) -> List[MultiModalMessage] | DialException:
    transformations = [
        await transform_message(file_storage, message) for message in messages
    ]
    all_errors = [
        error
        for error in transformations
        if isinstance(error, TransformationError)
    ]

    if all_errors:
        image_fails = set(
            image_fail
            for error in all_errors
            for image_fail in error.image_fails
        )
        msg = "The following file attachments failed to process:"
        msg += "\n".join(
            f"{idx}. {error.name}: {error.message}"
            for idx, error in enumerate(image_fails, start=1)
        )
        return InvalidRequestError(message=msg, display_message=msg)

    transformations = cast(List[MultiModalMessage], transformations)
    return transformations
