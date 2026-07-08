import base64

from aidial_sdk.chat_completion.request import Attachment

from aidial_adapter_openai.dial_api.storage import FileStorage


async def create_dial_attachment(
    *,
    file_storage: FileStorage | None,
    upload_dir: str,
    title: str,
    content_type: str,
    data: str | bytes,
) -> Attachment:
    if file_storage:
        metadata = await file_storage.upload_file(
            upload_dir, data, content_type
        )
        url = metadata.url
        data_b64 = None
    else:
        url = None
        data_b64 = (
            base64.b64encode(data).decode("utf-8")
            if isinstance(data, bytes)
            else data
        )

    return Attachment(title=title, type=content_type, url=url, data=data_b64)
