from aidial_adapter_openai.dial_api.storage import FileStorage


async def _upload_attachment_to_storage(
    file_storage: FileStorage, attachment: dict
):
    if (
        "data" not in attachment
        or "type" not in attachment
        or not attachment["type"].startswith("image/")
    ):
        return

    file_metadata = await file_storage.upload_file(
        "images", attachment["data"], attachment["type"]
    )

    del attachment["data"]
    attachment["url"] = file_metadata["url"]


async def upload_message_attachments_to_storage(
    file_storage: FileStorage | None, message: dict
):
    if (
        file_storage
        and (cc := message.get("custom_content"))
        and (attachments := cc.get("attachments"))
    ):
        for attachment in attachments:
            await _upload_attachment_to_storage(file_storage, attachment)
