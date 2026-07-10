import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from aidial_sdk.exceptions import RequestValidationError

# Only regular web schemes are allowed. Everything else (e.g. `file`,
# `ftp`, `gopher`) could be abused to reach local files or internal
# services.
_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _is_public_ip(ip: str) -> bool:
    address = ipaddress.ip_address(ip)

    # IPv4-mapped IPv6 addresses (e.g. `::ffff:169.254.169.254`) must be
    # judged by the semantics of the underlying IPv4 address. On Python < 3.13
    # `is_global` does not unwrap them automatically.
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            address = mapped

    return address.is_global


async def _resolve_host(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise RequestValidationError(
            f"Can't resolve the host of the file URL: {host}"
        ) from e
    return [info[4][0] for info in infos]


async def validate_public_url(url: str) -> None:
    """Guard against SSRF by rejecting file URLs that don't point to a
    publicly routable address.

    An attacker can supply an arbitrary attachment URL (e.g. the cloud
    metadata endpoint `http://169.254.169.254`) and force the adapter to
    fetch it. We only allow `http`/`https` URLs whose host resolves
    exclusively to globally routable IP addresses.
    """
    parsed = urlsplit(url)

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise RequestValidationError(
            f"Downloading files over the {scheme or 'empty'!r} URL scheme "
            "is not allowed"
        )

    host = parsed.hostname
    if not host:
        raise RequestValidationError("The file URL has no host")

    for ip in await _resolve_host(host):
        if not _is_public_ip(ip):
            raise RequestValidationError(
                "Downloading files from a non-public address "
                f"({ip}) is not allowed"
            )
