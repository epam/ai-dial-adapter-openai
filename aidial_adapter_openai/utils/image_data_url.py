import base64
import re
from typing import Optional

from pydantic import BaseModel


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
