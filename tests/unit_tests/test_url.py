import httpx
import pytest
import respx
from aidial_sdk.exceptions import RequestValidationError

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.url import (
    download_public_file,
    has_same_origin,
    validate_public_url,
)


@pytest.mark.parametrize(
    "url",
    [
        # Cloud metadata endpoint from the security report (link-local).
        "http://169.254.169.254/metadata/v1/instanceinfo",
        "http://127.0.0.1/",
        "http://localhost/secret",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",  # noqa: S104
        "https://[::1]/",
        "http://[::ffff:169.254.169.254]/",
        # Decimal-encoded 127.0.0.1.
        "http://2130706433/",
    ],
)
async def test_rejects_non_public_address(url: str):
    with pytest.raises(RequestValidationError, match="non-public address"):
        await validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://8.8.8.8/",
        "gopher://8.8.8.8/",
        "//8.8.8.8/",  # empty scheme
    ],
)
async def test_rejects_disallowed_scheme(url: str):
    with pytest.raises(RequestValidationError, match="scheme .*is not allowed"):
        await validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/file.txt",
        "https://1.1.1.1/file.txt",
        "https://[2606:4700:4700::1111]/file.txt",
    ],
)
async def test_allows_public_address(url: str):
    await validate_public_url(url)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("http://dial-core/", "http://dial-core"),
        ("http://dial-core/v1/files/x", "http://dial-core"),
        ("http://dial-core:80/x", "http://dial-core/"),
        ("https://dial-core:443/x", "https://dial-core/"),
    ],
)
def test_has_same_origin_true(a: str, b: str):
    assert has_same_origin(a, b) is True


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # `userinfo` trick: the real host is the metadata endpoint.
        ("http://dial-core@169.254.169.254/x", "http://dial-core"),
        # Look-alike host that merely starts with the DIAL URL string.
        ("http://dial-core.attacker.example/x", "http://dial-core"),
        # Different scheme / port.
        ("https://dial-core/x", "http://dial-core"),
        ("http://dial-core:8080/x", "http://dial-core"),
    ],
)
def test_has_same_origin_false(a: str, b: str):
    assert has_same_origin(a, b) is False


@respx.mock
async def test_download_public_file_allows_public_url():
    respx.get("http://8.8.8.8/image.png").mock(
        return_value=httpx.Response(200, content=b"public-bytes")
    )
    assert (
        await download_public_file("http://8.8.8.8/image.png")
        == b"public-bytes"
    )


async def test_download_public_file_blocks_internal_url():
    # Validation happens before any request is issued.
    with pytest.raises(RequestValidationError):
        await download_public_file("http://169.254.169.254/latest/meta-data/")


@respx.mock
async def test_download_public_file_follows_public_redirect():
    respx.get("http://8.8.8.8/a").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://1.1.1.1/b"}
        )
    )
    respx.get("http://1.1.1.1/b").mock(
        return_value=httpx.Response(200, content=b"redirected-bytes")
    )
    assert await download_public_file("http://8.8.8.8/a") == b"redirected-bytes"


@respx.mock
async def test_download_public_file_blocks_redirect_into_internal():
    # A public URL that redirects to an internal address must be rejected
    # when the redirect target is re-validated.
    respx.get("http://8.8.8.8/a").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://169.254.169.254/secret"}
        )
    )
    with pytest.raises(RequestValidationError):
        await download_public_file("http://8.8.8.8/a")


@respx.mock
async def test_download_public_file_rejects_redirect_loop():
    respx.get("http://8.8.8.8/loop").mock(
        return_value=httpx.Response(
            302, headers={"Location": "http://8.8.8.8/loop"}
        )
    )
    with pytest.raises(RequestValidationError, match="too many redirects"):
        await download_public_file("http://8.8.8.8/loop")


@pytest.mark.parametrize(
    "link",
    [
        # `userinfo` trick: prefix matches dial_url but the real host is
        # the cloud metadata endpoint.
        "http://dial-core@169.254.169.254/metadata/v1/instanceinfo",
        # A look-alike domain that merely starts with the dial_url string.
        "http://dial-core.attacker.example/10.0.0.1",
    ],
)
async def test_file_storage_does_not_trust_prefix_lookalikes(link: str):
    storage = FileStorage.create(dial_url="http://dial-core", api_key="secret")

    # A non-DIAL origin must be treated as untrusted and validated, so a
    # non-public target is rejected before any authenticated request is made.
    with pytest.raises(RequestValidationError):
        await storage.download_file(link)
