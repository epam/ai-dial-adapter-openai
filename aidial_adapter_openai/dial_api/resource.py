import base64
import mimetypes
from abc import ABC, abstractmethod
from typing import List

from aidial_sdk.chat_completion import Attachment
from pydantic import BaseModel, Field, root_validator

from aidial_adapter_openai.dial_api.storage import FileStorage, download_file
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.text import truncate_string


class ValidationError(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MissingContentTypeError(ValidationError):
    pass


class UnsupportedContentTypeError(ValidationError):
    type: str
    supported_types: List[str]

    def __init__(self, *, message: str, type: str, supported_types: List[str]):
        self.type = type
        self.supported_types = supported_types
        super().__init__(message)


class DialResource(ABC, BaseModel):
    entity_name: str = Field(default=None)
    supported_types: List[str] | None = Field(default=None)

    @abstractmethod
    async def download(self, storage: FileStorage | None) -> Resource: ...

    @abstractmethod
    async def guess_content_type(self) -> str | None: ...

    @abstractmethod
    async def get_resource_name(self, storage: FileStorage | None) -> str: ...

    async def get_content_type(self) -> str:
        type = await self.guess_content_type()

        if not type:
            raise MissingContentTypeError(
                f"Can't derive content type of the {self.entity_name}"
            )

        if (
            self.supported_types is not None
            and type not in self.supported_types
        ):
            raise UnsupportedContentTypeError(
                message=f"The {self.entity_name} is not one of the supported types",
                type=type,
                supported_types=self.supported_types,
            )

        return type


class URLResource(DialResource):
    url: str
    content_type: str | None = None

    @root_validator
    def validator(cls, values):
        values["entity_name"] = values.get("entity_name") or "URL"
        return values

    async def download(self, storage: FileStorage | None) -> Resource:
        type = await self.get_content_type()
        data = await _download_url(storage, self.url)
        return Resource(type=type, data=data)

    async def guess_content_type(self) -> str | None:
        return (
            self.content_type
            or Resource.parse_data_url_content_type(self.url)
            or mimetypes.guess_type(self.url)[0]
        )

    def is_data_url(self) -> bool:
        return Resource.parse_data_url_content_type(self.url) is not None

    async def get_resource_name(self, storage: FileStorage | None) -> str:
        if self.is_data_url():
            return f"data URL ({await self.guess_content_type()})"

        name = self.url
        if storage is not None:
            name = await storage.get_human_readable_name(self.url)

        return truncate_string(name, n=50)


def parse_attachment(attachment_dict: dict) -> Attachment:
    """
    The helper parses a dictionary to an attachment object.

    Unfortunately, DIAL SDK defaults a missing content type to
    `text/markdown`: https://github.com/epam/ai-dial-sdk/blob/2835107e950c89645a2b619fecba2518fa2d7bb1/aidial_sdk/chat_completion/request.py#L22

    This helper works around this issue.
    """
    attachment = Attachment.parse_obj(attachment_dict)
    if "type" not in attachment_dict:
        attachment.type = None
    return attachment


class AttachmentResource(DialResource):
    attachment: Attachment

    @root_validator(pre=True)
    def validator(cls, values):
        values["entity_name"] = values.get("entity_name") or "attachment"
        return values

    async def download(self, storage: FileStorage | None) -> Resource:
        type = await self.get_content_type()

        if self.attachment.data:
            data = base64.b64decode(self.attachment.data)
        elif self.attachment.url:
            data = await _download_url(storage, self.attachment.url)
        else:
            raise ValidationError(f"Invalid {self.entity_name}")

        return Resource(type=type, data=data)

    def create_url_resource(self, url: str) -> URLResource:
        return URLResource(
            url=url,
            content_type=self.effective_content_type,
            entity_name=self.entity_name,
        )

    @property
    def effective_content_type(self) -> str | None:
        if (
            self.attachment.type is None
            or "octet-stream" in self.attachment.type
        ):
            return None
        return self.attachment.type

    async def guess_content_type(self) -> str | None:
        if url := self.attachment.url:
            type = await self.create_url_resource(url).guess_content_type()
            if type:
                return type

        return self.attachment.type

    async def get_resource_name(self, storage: FileStorage | None) -> str:
        if title := self.attachment.title:
            return title

        if self.attachment.data:
            return f"data {self.entity_name}"
        elif url := self.attachment.url:
            return await self.create_url_resource(url).get_resource_name(
                storage
            )
        else:
            raise ValidationError(f"Invalid {self.entity_name}")


async def _download_url(file_storage: FileStorage | None, url: str) -> bytes:
    if (resource := Resource.from_data_url(url)) is not None:
        return resource.data

    if file_storage:
        return await file_storage.download_file(url)
    else:
        return await download_file(url)
