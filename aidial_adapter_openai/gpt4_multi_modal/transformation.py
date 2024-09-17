from typing import List, Optional, cast

from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel, root_validator

from aidial_adapter_openai.gpt4_multi_modal.download import (
    ImageFail,
    download_image,
)
from aidial_adapter_openai.gpt4_multi_modal.image_tokenizer import (
    tokenize_image,
)
from aidial_adapter_openai.gpt4_multi_modal.messages import (
    create_image_message,
    create_text_message,
)
from aidial_adapter_openai.utils.image_data_url import ImageDataURL
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import FileStorage
from aidial_adapter_openai.utils.tokens import Tokenizer


class MessageTransformResult(BaseModel):
    tokens: int
    message: dict


class TransformMessagesResult(BaseModel):
    error_message: Optional[str] = None
    transformations: Optional[List[MessageTransformResult]] = None

    @root_validator(pre=True)
    def validate_info(cls, values):
        if (
            values.get("error_message") is None
            and values.get("transformations") is None
        ):
            raise ValueError(
                "Either 'error_message' or 'transformations' must be provided."
            )
        return values


class TruncateErrorMessage(str):
    pass


async def transform_message(
    file_storage: Optional[FileStorage], message: dict, tokenizer: Tokenizer
) -> MessageTransformResult | List[ImageFail]:
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

    logger.debug(f"image tokens: {image_tokens}")

    # Create parts of the message if there are images
    if image_messages:
        message = {
            **message,
            "content": [create_text_message(content)] + image_messages,
        }

    return MessageTransformResult(
        tokens=sum(image_tokens)
        + tokenizer.calculate_tokens_per_message(message),
        message=message,
    )


async def transform_messages(
    file_storage: Optional[FileStorage],
    messages: List[dict],
    tokenizer: Tokenizer,
) -> List[MessageTransformResult] | InvalidRequestError:
    transformations = [
        await transform_message(file_storage, message, tokenizer)
        for message in messages
    ]
    all_errors = set(
        [error for error in transformations if isinstance(error, ImageFail)]
    )
    if all_errors:
        msg = "The following file attachments failed to process:"
        for idx, error in enumerate(all_errors, start=1):
            msg += f"\n{idx}. {error.name}: {error.message}"
        return InvalidRequestError(message=msg, display_message=msg)

    transformations = cast(List[MessageTransformResult], transformations)
    return transformations
