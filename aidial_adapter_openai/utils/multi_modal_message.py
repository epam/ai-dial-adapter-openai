from typing import List

from pydantic import BaseModel

from aidial_adapter_openai.utils.image import (
    ImageDataURL,
    ImageDetail,
    ImageMetadata,
)


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


class MultiModalMessage(BaseModel):
    image_metadatas: List[ImageMetadata]
    raw_message: dict
