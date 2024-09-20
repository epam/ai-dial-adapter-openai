import base64
import re
from io import BytesIO
from typing import Literal, Optional

from PIL import Image
from pydantic import BaseModel

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


class ImageDataURL(BaseModel):
    """
    Encoding of an image as a data URL.
    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
    """

    type: str
    data: str

    @classmethod
    def from_data_url(cls, data_uri: str) -> Optional["ImageDataURL"]:
        pattern = r"^data:image\/([^;]+);base64,(.+)$"
        match = re.match(pattern, data_uri)
        if match is None:
            return None
        data = match.group(2)

        try:
            base64.b64decode(data)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(
            type=f"image/{match.group(1)}",
            data=data,
        )

    def to_data_url(self) -> str:
        return f"data:{self.type};base64,{self.data}"

    def __repr__(self) -> str:
        return self.to_data_url()[:100] + "..."


class ImageMetadata(BaseModel):
    """
    Image metadata extracted from the image data URL.
    """

    image: ImageDataURL
    width: int
    height: int
    detail: DetailLevel

    @classmethod
    def from_image_data_url(cls, image: ImageDataURL) -> "ImageMetadata":
        image_data = base64.b64decode(image.data)
        with Image.open(BytesIO(image_data)) as img:
            width, height = img.size

        return cls(
            image=image,
            width=width,
            height=height,
            detail=resolve_detail_level(width, height, "auto"),
        )
