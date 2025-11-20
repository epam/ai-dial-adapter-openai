from openai.types.chat.chat_completion_content_part_param import File
from pydantic import BaseModel

from aidial_adapter_openai.utils.resource.base import Resource


class FileResource(BaseModel):
    name: str
    resource: Resource

    def to_content_part(self) -> File:
        return {
            "type": "file",
            "file": {
                "filename": self.name,
                "file_data": self.resource.to_data_url(),
            },
        }
