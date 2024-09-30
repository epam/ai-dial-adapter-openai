import base64
import re
from typing import Optional

from pydantic import BaseModel


class DataURL(BaseModel):
    """
    Encoding of an image as a data URL.
    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
    """

    type: str
    data: str

    @classmethod
    def from_data_url(cls, data_uri: str) -> Optional["DataURL"]:
        type = cls.parse_content_type(data_uri)
        if type is None:
            return None

        data = data_uri.removeprefix(cls._to_data_url_prefix(type))

        try:
            base64.b64decode(data)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(type=type, data=data)

    def to_data_url(self) -> str:
        return f"{self._to_data_url_prefix(self.type)}{self.data}"

    def __repr__(self) -> str:
        return self.to_data_url()[:100] + "..."

    @staticmethod
    def parse_content_type(data_uri: str) -> Optional[str]:
        pattern = r"^data:([^;]+);base64,"
        match = re.match(pattern, data_uri)
        return None if match is None else match.group(1)

    @staticmethod
    def _to_data_url_prefix(content_type: str) -> str:
        return f"data:{content_type};base64,"
