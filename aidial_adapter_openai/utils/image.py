import base64
from io import BytesIO
from typing import Literal

from PIL import Image
from pydantic import BaseModel

from aidial_adapter_openai.utils.data_url import DataURL

DetailLevel = Literal["low", "high"]
ImageDetail = DetailLevel | Literal["auto"]


def resolve_detail_level(
    width: int, height: int, detail: ImageDetail
) -> DetailLevel:
    match detail:
        case "auto":
            is_low = width <= 512 and height <= 512
            return "low" if is_low else "high"
        case "low":
            return "low"
        case "high":
            return "high"


class ImageMetadata(BaseModel):
    """
    Image metadata extracted from the image data URL.
    """

    image: DataURL
    width: int
    height: int
    detail: DetailLevel

    @classmethod
    def from_image_data_url(cls, image: DataURL) -> "ImageMetadata":
        image_data = base64.b64decode(image.data)
        with Image.open(BytesIO(image_data)) as img:
            width, height = img.size

        return cls(
            image=image,
            width=width,
            height=height,
            detail=resolve_detail_level(width, height, "auto"),
        )
