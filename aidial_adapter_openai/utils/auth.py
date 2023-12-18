from typing import Mapping, Optional

from pydantic import BaseModel


class Auth(BaseModel):
    name: str
    value: str

    @property
    def headers(self) -> dict[str, str]:
        return {self.name: self.value}

    @classmethod
    def from_headers(
        cls, name: str, headers: Mapping[str, str]
    ) -> Optional["Auth"]:
        value = headers.get(name)
        if value is None:
            return None
        return cls(name=name, value=value)
