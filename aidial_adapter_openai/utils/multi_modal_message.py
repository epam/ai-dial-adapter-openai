from openai.types.chat import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_content_part_param import File
from pydantic import BaseModel

from aidial_adapter_openai.utils.resource.audio import AudioResource
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.file import FileResource
from aidial_adapter_openai.utils.resource.image import ImageResource


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
    images: list[ImageResource] = []
    files: list[FileResource] = []
    audios: list[AudioResource] = []
    raw_message: dict
