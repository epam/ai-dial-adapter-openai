from typing import List, Optional, cast

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
    text_tokens: int = 0
    image_tokens: int = 0
    errors: Optional[List[ImageFail]] = None
    message: dict

    @property
    def total_tokens(self) -> int:
        return self.text_tokens + self.image_tokens


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


class TruncateTransformedMessagesResult(BaseModel):
    messages: List[dict]
    discarded_messages: List[int]
    overall_token: int


async def transform_message(
    file_storage: Optional[FileStorage], message: dict, tokenizer: Tokenizer
) -> MessageTransformResult:
    content = message.get("content", "")
    custom_content = message.get("custom_content", {})
    attachments = custom_content.get("attachments", [])

    message = {k: v for k, v in message.items() if k != "custom_content"}

    if len(attachments) == 0:
        return MessageTransformResult(
            message=message,
            text_tokens=tokenizer.calculate_tokens_per_message(message),
        )

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
        return MessageTransformResult(message=message, errors=errors)

    image_urls: List[ImageDataURL] = cast(List[ImageDataURL], download_results)

    image_tokens: List[int] = []
    image_messages: List[dict] = []

    for image_url in image_urls:
        tokens, detail = tokenize_image(image_url, "auto")
        image_tokens.append(tokens)
        image_messages.append(create_image_message(image_url, detail))

    logger.debug(f"image tokens: {image_tokens}")

    return MessageTransformResult(
        image_tokens=sum(image_tokens),
        text_tokens=tokenizer.calculate_tokens_per_message(
            create_text_message(content)
        ),
        message={
            **message,
            "content": [create_text_message(content)] + image_messages,
        },
    )


async def transform_messages(
    file_storage: Optional[FileStorage],
    messages: List[dict],
    tokenizer: Tokenizer,
) -> TransformMessagesResult:
    transformations = [
        await transform_message(file_storage, message, tokenizer)
        for message in messages
    ]
    all_errors = set(
        [error for t in transformations if t.errors for error in t.errors]
    )
    if all_errors:
        msg = "The following file attachments failed to process:"
        for idx, error in enumerate(all_errors, start=1):
            msg += f"\n{idx}. {error.name}: {error.message}"
        return TransformMessagesResult(error_message=msg)

    return TransformMessagesResult(transformations=transformations)
