import pytest
from aidial_sdk.exceptions import RequestValidationError

from aidial_adapter_openai.dial_api._ssrf import validate_public_url
from aidial_adapter_openai.dial_api.storage import FileStorage


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
