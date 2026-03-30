from dataclasses import dataclass
from typing import List, Set, assert_never

from aidial_sdk.exceptions import InvalidRequestError
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_param import File
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
    ValidationError,
    parse_attachment,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import (
    MultiModalMessage,
    create_text_content_part,
)
from aidial_adapter_openai.utils.resource.audio import (
    AudioResource,
    AudioUrlContentPart,
)
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.file import FileResource
from aidial_adapter_openai.utils.resource.image import ImageResource
from aidial_adapter_openai.utils.text import decapitalize
from aidial_adapter_openai.utils.validation import (
    ensure_dict,
    ensure_list,
    ensure_list_or_str,
    ensure_str,
    ensure_str_or_none,
)


@dataclass(order=True, frozen=True)
class Error:
    name: str
    message: str


class MessageTransformer:
    file_storage: FileStorage | None
    errors: Set[Error]
    images: List[ImageResource]
    files: List[FileResource]
    audios: List[AudioResource]
    support_audio: bool

    def __init__(
        self,
        *,
        file_storage: FileStorage | None,
        errors: Set[Error] | None = None,
        support_audio: bool = False,
    ):
        self.file_storage = file_storage
        self.errors = set() if errors is None else errors
        self.images = []
        self.files = []
        self.audios = []
        self.support_audio = support_audio

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
    ) -> List[
        ChatCompletionContentPartImageParam | AudioUrlContentPart | File
    ]:
        if attachments:
            logger.debug(f"original attachments: {attachments}")

        ret: List[
            ChatCompletionContentPartImageParam | AudioUrlContentPart | File
        ] = []
        for attachment in attachments:
            if result := await self.download_attachment(attachment):
                ret.append(result)
        return ret

    async def download_attachment(
        self, attachment: dict
    ) -> (
        ChatCompletionContentPartImageParam
        | AudioUrlContentPart
        | File
        | None
    ):
        dial_resource = AttachmentResource(
            attachment=parse_attachment(attachment),
            entity_name="attachment",
        )

        if not (resource := await self.try_download_resource(dial_resource)):
            return None

        content_type = await dial_resource.guess_content_type()
        if content_type and content_type.startswith("image/"):
            result = await ImageResource.from_resource(resource, None)
            self.images.append(result)
        elif (
            self.support_audio
            and content_type
            and content_type.startswith("audio/")
        ):
            result = AudioResource.from_resource(resource)
            self.audios.append(result)
        else:
            name = await dial_resource.get_resource_name(self.file_storage)
            result = FileResource(name=name, resource=resource)
            self.files.append(result)

        return result.to_content_part()

    async def download_image_content_part(
        self, part: ChatCompletionContentPartImageParam
    ) -> ChatCompletionContentPartImageParam | None:
        image_url = ensure_dict("image_url", part["image_url"])
        detail = ensure_str_or_none("image_url.detail", image_url.get("detail"))
        url = ensure_str("image_url.url", image_url.get("url"))

        dial_resource = URLResource(url=url, entity_name="image content part")
        if not (resource := await self.try_download_resource(dial_resource)):
            return None

        result = await ImageResource.from_resource(resource, detail)  # type: ignore
        self.images.append(result)
        return result.to_content_part()

    def _convert_input_audio_to_audio_url(
        self, part: dict
    ) -> AudioUrlContentPart:
        """Convert an OpenAI ``input_audio`` content part to a vLLM ``audio_url`` part."""
        input_audio = ensure_dict("input_audio", part.get("input_audio"))
        data = ensure_str("input_audio.data", input_audio.get("data"))
        fmt = ensure_str("input_audio.format", input_audio.get("format"))
        mime = f"audio/{fmt}"
        resource = Resource.from_base64(type=mime, data_base64=data)
        result = AudioResource(audio=resource)
        self.audios.append(result)
        return result.to_content_part()

    async def download_content_part(
        self, part: ChatCompletionContentPartParam | ContentArrayOfContentPart
    ) -> ChatCompletionContentPartParam | ContentArrayOfContentPart | None:
        ensure_dict("content part", part)
        match part["type"]:
            case "image_url":
                return await self.download_image_content_part(part)
            case "input_audio":
                if self.support_audio:
                    return self._convert_input_audio_to_audio_url(part)
                return part
            case "text" | "refusal" | "audio_url" | "file":
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
        for part in parts:
            if result := await self.download_content_part(part):
                ret.append(result)
        return ret

    async def transform_message(self, message: dict) -> MultiModalMessage:
        message = ensure_dict("message", message).copy()

        content = ensure_list_or_str("content", message.get("content") or "")
        custom_content = ensure_dict(
            "custom_content", message.pop("custom_content", None) or {}
        )
        attachments = ensure_list(
            "attachments", custom_content.get("attachments") or []
        )

        if isinstance(content, str) and not attachments:
            return MultiModalMessage(raw_message=message)

        content_parts = await self.download_content(content)
        attachment_parts = await self.download_attachments(attachments)

        return MultiModalMessage(
            images=self.images,
            files=self.files,
            audios=self.audios,
            raw_message={
                **message,
                "content": content_parts + attachment_parts,
            },
        )


class ResourceProcessor(BaseModel):
    file_storage: FileStorage | None
    support_audio: bool = False

    async def transform_messages(
        self, messages: List[dict]
    ) -> List[MultiModalMessage]:
        errors: Set[Error] = set()
        transformations = [
            await MessageTransformer(
                file_storage=self.file_storage,
                errors=errors,
                support_audio=self.support_audio,
            ).transform_message(message)
            for message in messages
        ]

        if errors:
            fails = sorted(errors)
            msg = "The following files failed to process:\n"
            msg += "\n".join(
                f"{idx}. {error.name}: {decapitalize(error.message)}"
                for idx, error in enumerate(fails, start=1)
            )
            raise InvalidRequestError(message=msg, display_message=msg)

        return transformations
