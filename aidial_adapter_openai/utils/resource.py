import base64
import re
from typing import Optional

from pydantic import BaseModel


class Resource(BaseModel):
    type: str
    data: bytes

    @classmethod
    def from_base64(cls, type: str, data_base64: str) -> "Resource":
        try:
            data = base64.b64decode(data_base64, validate=True)
        except Exception:
            raise ValueError("Invalid base64 data")

        return cls(type=type, data=data)

    @classmethod
    def from_data_url(cls, data_url: str) -> Optional["Resource"]:
        """
        Parsing a resource encoded as a data URL.
        See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs for reference.
        """

        type = cls.parse_data_url_content_type(data_url)
        if type is None:
            return None

        data_base64 = data_url.removeprefix(cls._to_data_url_prefix(type))

        return cls.from_base64(type, data_base64)

    @property
    def data_base64(self) -> str:
        return base64.b64encode(self.data).decode()

    def to_data_url(self) -> str:
        return f"{self._to_data_url_prefix(self.type)}{self.data_base64}"

    @staticmethod
    def parse_data_url_content_type(data_url: str) -> Optional[str]:
        pattern = r"^data:([^;]+);base64,"
        match = re.match(pattern, data_url)
        return None if match is None else match.group(1)

    @staticmethod
    def _to_data_url_prefix(content_type: str) -> str:
        return f"data:{content_type};base64,"

    def __str__(self) -> str:
        return self.to_data_url()[:100] + "..."
