from io import BytesIO
from typing import Literal, assert_never

from PIL import Image
from pydantic import BaseModel

from aidial_adapter_openai.utils.resource import Resource

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
        case _:
            assert_never(detail)


class ImageMetadata(BaseModel):
    """
    Image metadata extracted from the image data URL.
    """

    image: Resource
    width: int
    height: int
    detail: DetailLevel

    @classmethod
    def from_resource(cls, image: Resource) -> "ImageMetadata":
        with Image.open(BytesIO(image.data)) as img:
            width, height = img.size

        return cls(
            image=image,
            width=width,
            height=height,
            detail=resolve_detail_level(width, height, "auto"),
        )
