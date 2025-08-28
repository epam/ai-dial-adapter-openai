from dataclasses import dataclass
from typing import List, Set, assert_never, cast

from aidial_sdk.exceptions import InvalidRequestError
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_param import File
from pydantic import BaseModel, Field

from aidial_adapter_openai.dial_api.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
    ValidationError,
    parse_attachment,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.image import ImageResource
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import (
    MultiModalMessage,
    create_text_content_part,
)
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.text import decapitalize
from aidial_adapter_openai.utils.validation import RequestValidationMixin


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


@dataclass(order=True, frozen=True)
class Error:
    name: str
    message: str


class ResourceProcessor(RequestValidationMixin, BaseModel):
    class Config:
        arbitrary_types_allowed = True  # for errors

    file_storage: FileStorage | None

    errors: Set[Error] = Field(default_factory=set)
    images: List[ImageResource] = []

    async def try_download_resource(
        self, dial_resource: DialResource
    ) -> Resource | None:
        try:
            return await dial_resource.download(self.file_storage)
        except Exception as e:
            logger.error(
                f"Failed to download {dial_resource.entity_name}: {str(e)}"
            )

            name = await dial_resource.get_resource_name(self.file_storage)
            message = (
                e.message
                if isinstance(e, ValidationError)
                else f"Failed to download the {dial_resource.entity_name}"
            )
            self.errors.add(Error(name=name, message=message))
            return None

    async def download_attachments(
        self, attachments: List[dict]
    ) -> List[ChatCompletionContentPartImageParam | File]:

        if attachments:
            logger.debug(f"original attachments: {attachments}")

        ret: List[ChatCompletionContentPartImageParam | File] = []
        for idx, attachment in enumerate(attachments):
            with self.path_(idx):
                if result := await self.download_attachment(attachment):
                    ret.append(result)
        return ret

    async def download_attachment(
        self, attachment: dict
    ) -> ChatCompletionContentPartImageParam | File | None:
        dial_resource = AttachmentResource(
            attachment=parse_attachment(attachment),
            entity_name="attachment",
        )

        if not (resource := await self.try_download_resource(dial_resource)):
            return None

        content_type = await dial_resource.guess_content_type()
        if content_type and content_type.startswith("image/"):
            result = ImageResource.from_resource(resource, None)
            self.images.append(result)
        else:
            name = await dial_resource.get_resource_name(self.file_storage)
            result = FileResource(name=name, resource=resource)

        return result.to_content_part()

    async def download_image_content_part(
        self, part: ChatCompletionContentPartImageParam
    ) -> ChatCompletionContentPartImageParam | None:
        image_url = part["image_url"]
        detail = image_url.get("detail")
        url = image_url.get("url")

        dial_resource = URLResource(url=url, entity_name="image content part")
        if not (resource := await self.try_download_resource(dial_resource)):
            return None

        result = ImageResource.from_resource(resource, detail)
        self.images.append(result)
        return result.to_content_part()

    async def download_content_part(
        self, part: ChatCompletionContentPartParam | ContentArrayOfContentPart
    ) -> ChatCompletionContentPartParam | ContentArrayOfContentPart | None:
        match part["type"]:
            case "image_url":
                return await self.download_image_content_part(part)
            case "text" | "refusal" | "input_audio" | "file":
                return part
            case _:
                assert_never(part)

    async def download_content(
        self,
        content: (
            str
            | List[ChatCompletionContentPartParam | ContentArrayOfContentPart]
        ),
    ) -> List[ChatCompletionContentPartParam | ContentArrayOfContentPart]:
        if isinstance(content, str):
            parts = [create_text_content_part(content)]
        else:
            parts = content

        ret: List[
            ChatCompletionContentPartParam | ContentArrayOfContentPart
        ] = []
        for idx, part in enumerate(parts):
            with self.path_(idx):
                if result := await self.download_content_part(part):
                    ret.append(result)
        return ret

    async def transform_message(self, message: dict) -> MultiModalMessage:
        message = message.copy()

        content = message.get("content") or ""
        custom_content = message.pop("custom_content", None) or {}
        attachments: List[dict] = custom_content.get("attachments") or []

        if isinstance(content, str) and not attachments:
            return MultiModalMessage(images=[], raw_message=message)

        with self.path_("content"):
            content_parts = await self.download_content(content)

        with self.path_("custom_content", "attachments"):
            attachment_parts = await self.download_attachments(attachments)

        return MultiModalMessage(
            images=self.images,
            raw_message={
                **message,
                "content": content_parts + attachment_parts,
            },
        )

    async def transform_messages(
        self, messages: List[dict]
    ) -> List[MultiModalMessage]:
        transformations: List[MultiModalMessage] = []
        for idx, message in enumerate(messages):
            with self.path_("messages", idx):
                transformations.append(await self.transform_message(message))

        if self.errors:
            image_fails = sorted(list(self.errors))
            msg = "The following files failed to process:\n"
            msg += "\n".join(
                f"{idx}. {error.name}: {decapitalize(error.message)}"
                for idx, error in enumerate(image_fails, start=1)
            )
            raise InvalidRequestError(message=msg, display_message=msg)

        transformations = cast(List[MultiModalMessage], transformations)
        return transformations
