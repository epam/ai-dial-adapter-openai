from typing import List

from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_content_part_param import File
from pydantic import BaseModel

from aidial_adapter_openai.utils.image import ImageResource
from aidial_adapter_openai.utils.resource import Resource


def create_file_content_part(filename: str, resource: Resource) -> File:
    return {
        "type": "file",
        "file": {"file_data": resource.to_data_url(), "filename": filename},
    }


def create_text_content_part(text: str) -> ChatCompletionContentPartTextParam:
    return {
        "type": "text",
        "text": text,
    }


class MultiModalMessage(BaseModel):
    images: List[ImageResource]
    raw_message: dict
