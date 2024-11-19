from dataclasses import dataclass
from typing import List, Optional, Set, cast

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel, Field

from aidial_adapter_openai.dial_api.resource import (
    AttachmentResource,
    DialResource,
    URLResource,
    ValidationError,
    parse_attachment,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.image import ImageDetail, ImageMetadata
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import (
    MultiModalMessage,
    create_image_content_part,
    create_text_content_part,
)
from aidial_adapter_openai.utils.resource import Resource
from aidial_adapter_openai.utils.text import decapitalize

# Officially supported image types by GPT-4 Vision, GPT-4o
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"]


@dataclass(order=True, frozen=True)
class TransformationError:
    name: str
    message: str


class ResourceProcessor(BaseModel):
    class Config:
        arbitrary_types_allowed = True  # for errors

    file_storage: FileStorage | None
    errors: Set[TransformationError] = Field(default_factory=set)

    def collect_resource(
        self,
        meta: List[ImageMetadata],
        result: Resource | TransformationError,
        detail: Optional[ImageDetail],
    ):
        if isinstance(result, TransformationError):
            self.errors.add(result)
        else:
            meta.append(ImageMetadata.from_resource(result, detail))

    async def try_download_resource(
        self, dial_resource: DialResource
    ) -> Resource | TransformationError:
        try:
            resource = await dial_resource.download(self.file_storage)
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
            return TransformationError(name=name, message=message)

        return resource

    async def download_attachment_images(
        self, attachments: List[dict]
    ) -> List[ImageMetadata]:
        if attachments:
            logger.debug(f"original attachments: {attachments}")

        ret: List[ImageMetadata] = []

        for attachment in attachments:
            dial_resource = AttachmentResource(
                attachment=parse_attachment(attachment),
                entity_name="image attachment",
                supported_types=SUPPORTED_IMAGE_TYPES,
            )
            result = await self.try_download_resource(dial_resource)
            self.collect_resource(ret, result, None)

        return ret

    async def download_content_images(
        self, content: str | list
    ) -> List[ImageMetadata]:
        if isinstance(content, str):
            return []

        ret: List[ImageMetadata] = []

        for content_part in content:
            image_url = content_part.get("image_url")
            if image_url and (url := image_url.get("url")):
                detail = image_url.get("detail")
                if detail not in [None, "auto", "low", "high"]:
                    raise ValidationError("Unexpected image detail")

                dial_resource = URLResource(
                    url=url,
                    entity_name="image",
                    supported_types=SUPPORTED_IMAGE_TYPES,
                )
                result = await self.try_download_resource(dial_resource)
                self.collect_resource(ret, result, detail)

        return ret

    async def transform_message(self, message: dict) -> MultiModalMessage:
        message = message.copy()

        content = message.get("content") or ""
        custom_content = message.pop("custom_content", None) or {}
        attachments = custom_content.get("attachments") or []

        attachment_meta = await self.download_attachment_images(attachments)
        content_meta = await self.download_content_images(content)
        meta = [*content_meta, *attachment_meta]

        if not meta:
            return MultiModalMessage(image_metadatas=[], raw_message=message)

        content_parts = (
            [create_text_content_part(content)]
            if isinstance(content, str)
            else content
        ) + [
            create_image_content_part(meta.image, meta.detail)
            for meta in attachment_meta
        ]

        return MultiModalMessage(
            image_metadatas=meta,
            raw_message={**message, "content": content_parts},
        )

    async def transform_messages(
        self, messages: List[dict]
    ) -> List[MultiModalMessage] | DialException:
        transformations = [
            await self.transform_message(message) for message in messages
        ]

        if self.errors:
            image_fails = sorted(list(self.errors))
            msg = "The following files failed to process:\n"
            msg += "\n".join(
                f"{idx}. {error.name}: {decapitalize(error.message)}"
                for idx, error in enumerate(image_fails, start=1)
            )
            return InvalidRequestError(message=msg, display_message=msg)

        transformations = cast(List[MultiModalMessage], transformations)
        return transformations
