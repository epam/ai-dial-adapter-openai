"""
Module for creating parts of "content" field in a message.
Used to create a message with text and image content parts.
"""

from typing import Literal

from aidial_adapter_openai.utils.image_data_url import ImageDataURL

DetailLevel = Literal["low", "high"]
ImageDetail = DetailLevel | Literal["auto"]


def create_image_content_part(image: ImageDataURL, detail: ImageDetail) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": image.to_data_url(),
            "detail": detail,
        },
    }


def create_text_content_part(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
    }
