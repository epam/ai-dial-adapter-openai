from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from aidial_client import DialException
from aidial_client.types.metadata import FileMetadata
from aidial_sdk.exceptions import InvalidRequestError, RequestValidationError

from aidial_adapter_openai.dial_api.storage import FileStorage, download_file


def _make_storage() -> FileStorage:
    return FileStorage.create(dial_url="http://dial-core", api_key="test-key")


def _make_dial_client(
    *,
    appdata_home: PurePosixPath | None = None,
    files_home: PurePosixPath | None = None,
    upload_result: FileMetadata | None = None,
    download_result: bytes = b"from-sdk",
    download_error: Exception | None = None,
):
    if appdata_home is None:
        appdata_home = PurePosixPath("user-bucket/appdata/test-app")
    if files_home is None:
        files_home = PurePosixPath("files/user-bucket")
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
        my_appdata_home=AsyncMock(return_value=appdata_home),
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
async def test_upload_raises_when_appdata_unavailable(monkeypatch):
    storage = _make_storage()
    dial_client = _make_dial_client()
    dial_client.my_appdata_home = AsyncMock(return_value=None)
    monkeypatch.setattr(storage, "client", dial_client)

    with pytest.raises(
        ValueError, match="Unable to retrieve user appdata directory"
    ):
        await storage.upload(
            upload_dir="images",
            filename="sha256",
            content_type="image/png",
            content=b"binary-content",
        )

    dial_client.files.upload.assert_not_awaited()


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

    result = await _make_storage().download_file("http://test/image.png")
    assert result == b"from-raw-http"
    assert captured["url"] == "http://test/image.png"


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


@pytest.mark.asyncio
@respx.mock
async def test_raw_download_allows_public_url():
    respx.get("http://8.8.8.8/image.png").mock(
        return_value=httpx.Response(200, content=b"public-bytes")
    )
    assert await download_file("http://8.8.8.8/image.png") == b"public-bytes"


@pytest.mark.asyncio
async def test_raw_download_blocks_internal_url():
    # Validation happens before any request is issued.
    with pytest.raises(RequestValidationError):
        await download_file("http://169.254.169.254/latest/meta-data/")


@pytest.mark.asyncio
@respx.mock
async def test_raw_download_follows_public_redirect():
    respx.get("http://8.8.8.8/a").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://1.1.1.1/b"}
        )
    )
    respx.get("http://1.1.1.1/b").mock(
        return_value=httpx.Response(200, content=b"redirected-bytes")
    )
    assert await download_file("http://8.8.8.8/a") == b"redirected-bytes"


@pytest.mark.asyncio
@respx.mock
async def test_raw_download_blocks_redirect_into_internal():
    # A public URL that redirects to an internal address must be rejected
    # when the redirect target is re-validated.
    respx.get("http://8.8.8.8/a").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://169.254.169.254/secret"}
        )
    )
    with pytest.raises(RequestValidationError):
        await download_file("http://8.8.8.8/a")


@pytest.mark.asyncio
@respx.mock
async def test_raw_download_rejects_redirect_loop():
    respx.get("http://8.8.8.8/loop").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://8.8.8.8/loop"}
        )
    )
    with pytest.raises(RequestValidationError, match="too many redirects"):
        await download_file("http://8.8.8.8/loop")
