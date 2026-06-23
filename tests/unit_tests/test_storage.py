from pathlib import PurePosixPath
from typing import Any

import pytest
from aidial_client import DialException
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import SecretStr

from aidial_adapter_openai.dial_api.storage import FileStorage


class _FakeMetadata:
    def model_dump(self) -> dict[str, Any]:
        return {
            "name": "sha256.png",
            "parent_path": "images",
            "bucket": "user-bucket",
            "url": "files/user-bucket/images/sha256.png",
        }


class _FakeDownloadResult:
    async def aget_content(self) -> bytes:
        return b"from-sdk"


@pytest.mark.asyncio
async def test_upload_uses_dial_client_sdk(monkeypatch):
    calls: dict[str, Any] = {}

    class _FakeFiles:
        async def upload(self, *, url, file):
            calls["url"] = str(url)
            calls["file"] = file
            return _FakeMetadata()

    class _FakeDialClient:
        files = _FakeFiles()

        async def my_files_home(self):
            return PurePosixPath("files/user-bucket/appdata/test-app")

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.AsyncDial",
        lambda **_: _FakeDialClient(),
    )

    storage = FileStorage(
        dial_url="http://dial-core",
        api_key=SecretStr("test-key"),
    )

    metadata = await storage.upload(
        upload_dir="images",
        filename="sha256",
        content_type="image/png",
        content=b"binary-content",
    )

    assert (
        calls["url"] == "files/user-bucket/appdata/test-app/images/sha256.png"
    )
    assert calls["file"] == ("sha256.png", b"binary-content", "image/png")
    assert metadata == {
        "name": "sha256.png",
        "parentPath": "images",
        "bucket": "user-bucket",
        "url": "files/user-bucket/images/sha256.png",
    }


@pytest.mark.asyncio
async def test_download_dial_files_url_uses_sdk(monkeypatch):
    class _FakeFiles:
        async def download(self, *, url):
            assert (
                url == "http://dial-core/v1/files/user-bucket/images/sample.png"
            )
            return _FakeDownloadResult()

    class _FakeDialClient:
        files = _FakeFiles()

    def _unexpected_raw_download(*args, **kwargs):
        raise AssertionError("raw download should not be called")

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.AsyncDial",
        lambda **_: _FakeDialClient(),
    )
    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.download_file",
        _unexpected_raw_download,
    )

    storage = FileStorage(
        dial_url="http://dial-core",
        api_key=SecretStr("test-key"),
    )

    result = await storage.download_file("files/user-bucket/images/sample.png")
    assert result == b"from-sdk"


@pytest.mark.asyncio
async def test_download_non_file_dial_url_uses_raw_http(monkeypatch):
    captured: dict[str, Any] = {}

    async def _fake_download_file(url: str, headers):
        captured["url"] = url
        captured["headers"] = headers
        return b"from-raw-http"

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.download_file",
        _fake_download_file,
    )

    storage = FileStorage(
        dial_url="http://dial-core",
        api_key=SecretStr("test-key"),
    )

    result = await storage.download_file("images/sample.png")
    assert result == b"from-raw-http"
    assert captured["url"] == "http://dial-core/v1/images/sample.png"
    assert captured["headers"] == {"api-key": "test-key"}


@pytest.mark.asyncio
async def test_download_sdk_errors_are_mapped_to_invalid_request(monkeypatch):
    class _FakeFiles:
        async def download(self, *, url):
            raise DialException("denied", 403)

    class _FakeDialClient:
        files = _FakeFiles()

    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.AsyncDial",
        lambda **_: _FakeDialClient(),
    )

    storage = FileStorage(
        dial_url="http://dial-core",
        api_key=SecretStr("test-key"),
    )

    with pytest.raises(InvalidRequestError) as exc:
        await storage.download_file("files/user-bucket/images/sample.png")

    assert str(exc.value) == (
        "Failed to download file 'files/user-bucket/images/sample.png' "
        "(status code 403)"
    )
