from typing import Literal

from aidial_adapter_openai.utils.image_data_url import ImageDataURL

ConcreteImageDetail = Literal["low", "high"]
ImageDetail = ConcreteImageDetail | Literal["auto"]


def create_image_message(
    image: ImageDataURL, detail: ConcreteImageDetail
) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": image.to_data_url(),
            "detail": detail,
        },
    }


def create_text_message(text: str) -> dict:
    return {
        "type": "text",
        "text": text,
    }
