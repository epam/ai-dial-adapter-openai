from typing import List, Optional, cast

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.gpt4_multi_modal.attachment import (
    ImageFail,
    download_image,
)
from aidial_adapter_openai.utils.image_data_url import ImageDataURL
from aidial_adapter_openai.utils.image_tokenizer import tokenize_image
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.message_content_part import (
    create_image_content_part,
    create_text_content_part,
)
from aidial_adapter_openai.utils.storage import FileStorage
from aidial_adapter_openai.utils.tokenizer import Tokenizer


class MultiModalMessage(BaseModel):
    tokens: int
    message: dict


class TransformationError(BaseModel):
    image_fails: List[ImageFail]


async def transform_message(
    file_storage: Optional[FileStorage], message: dict, tokenizer: Tokenizer
) -> dict | TransformationError:
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

    return {
        **message,
        "content": [create_text_content_part(content)]
        + [
            create_image_content_part(image_url, "auto")
            for image_url in image_urls
        ],
    }


async def transform_messages(
    file_storage: Optional[FileStorage],
    messages: List[dict],
    tokenizer: Tokenizer,
) -> List[dict] | DialException:
    transformations = [
        await transform_message(file_storage, message, tokenizer)
        for message in messages
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

    transformations = cast(List[dict], transformations)
    return transformations
