from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aidial_client import DialException
from aidial_client.types.metadata import FileMetadata
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.dial_api.storage import FileStorage


def _make_storage() -> FileStorage:
    return FileStorage.create(dial_url="http://dial-core", api_key="test-key")


def _make_dial_client(
    *,
    files_home: PurePosixPath | None = None,
    upload_result: FileMetadata | None = None,
    download_result: bytes = b"from-sdk",
    download_error: Exception | None = None,
):
    if files_home is None:
        files_home = PurePosixPath("files/user-bucket/appdata/test-app")
    files = SimpleNamespace()
    files.upload = AsyncMock(return_value=upload_result)
    files.download = AsyncMock(
        side_effect=download_error
        if download_error is not None
        else [
            SimpleNamespace(
                aget_content=AsyncMock(return_value=download_result)
            )
        ]
    )
    return SimpleNamespace(
        base_url="http://dial-core/",
        is_dial_url=lambda _: True,
        auth_headers=AsyncMock(return_value={"api-key": "test-key"}),
        my_files_home=AsyncMock(return_value=files_home),
        files=files,
    )


@pytest.mark.asyncio
async def test_upload_uses_dial_client_sdk(monkeypatch):
    metadata = FileMetadata(
        name="sha256.png",
        parent_path="images",
        bucket="user-bucket",
        url="files/user-bucket/images/sha256.png",
        node_type="ITEM",
        resource_type="FILE",
    )
    storage = _make_storage()
    dial_client = _make_dial_client(upload_result=metadata)
    monkeypatch.setattr(storage, "client", dial_client)

    result = await storage.upload(
        upload_dir="images",
        filename="sha256",
        content_type="image/png",
        content=b"binary-content",
    )

    dial_client.files.upload.assert_awaited_once_with(
        url=PurePosixPath(
            "files/user-bucket/appdata/test-app/images/sha256.png"
        ),
        file=("sha256.png", b"binary-content", "image/png"),
    )
    assert result == metadata


@pytest.mark.asyncio
async def test_download_dial_files_url_uses_sdk(monkeypatch):
    storage = _make_storage()
    dial_client = _make_dial_client()
    monkeypatch.setattr(storage, "client", dial_client)

    async def _unexpected_raw_download(*args, **kwargs):
        raise AssertionError("raw download should not be called")

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.download_file",
        _unexpected_raw_download,
    )

    result = await storage.download_file("files/user-bucket/images/sample.png")
    dial_client.files.download.assert_awaited_once_with(
        url="files/user-bucket/images/sample.png"
    )
    assert result == b"from-sdk"


@pytest.mark.asyncio
async def test_download_non_dial_file_url_uses_raw_http(monkeypatch):
    captured: dict[str, str] = {}

    async def _fake_download_file(url: str):
        captured["url"] = url
        return b"from-raw-http"

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.download_file",
        _fake_download_file,
    )

    result = await _make_storage().download_file("images/sample.png")
    assert result == b"from-raw-http"
    assert captured["url"] == "images/sample.png"


@pytest.mark.asyncio
async def test_download_sdk_errors_are_mapped_to_invalid_request(monkeypatch):
    storage = _make_storage()
    dial_client = _make_dial_client(download_error=DialException("denied", 403))
    monkeypatch.setattr(storage, "client", dial_client)

    with pytest.raises(InvalidRequestError) as exc:
        await storage.download_file("files/user-bucket/images/sample.png")

    assert str(exc.value) == (
        "Failed to download file 'files/user-bucket/images/sample.png' "
        "(status code 403)"
    )
