from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings.request import EmbeddingsRequest

from aidial_adapter_openai.dial_api.embedding_inputs import (
    collect_embedding_inputs,
)
from aidial_adapter_openai.dial_api.resource import AttachmentResource
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.resource.base import Resource


async def resolve_embedding_inputs(
    request: dict,
    file_storage: FileStorage | None,
) -> list[str | Resource]:
    body = EmbeddingsRequest.model_validate(request)

    async def on_text(text: str) -> str:
        return text

    async def on_attachment(attachment: Attachment) -> Resource:
        return await AttachmentResource(attachment=attachment).download(
            file_storage
        )

    return [
        item
        async for item in collect_embedding_inputs(
            body,
            on_text=on_text,
            on_attachment=on_attachment,
        )
    ]
