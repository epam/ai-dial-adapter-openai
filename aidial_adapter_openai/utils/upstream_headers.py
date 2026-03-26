from typing import Mapping

from aidial_sdk.exceptions import HTTPException
from pydantic_core import from_json


def _bad_upstream_extra_data(message: str) -> HTTPException:
    return HTTPException(
        status_code=502,
        type="internal_server_error",
        message=f"Invalid X-UPSTREAM-EXTRA-DATA header: {message}",
    )


def get_upstream_extra_headers(
    request_headers: Mapping[str, str],
) -> dict[str, str]:
    """Extract upstream headers listed in X-UPSTREAM-EXTRA-DATA."""
    extra_data_header = request_headers.get("X-UPSTREAM-EXTRA-DATA")
    if not extra_data_header:
        return {}

    try:
        extra_data = from_json(extra_data_header)
    except ValueError as e:
        raise _bad_upstream_extra_data(f"JSON parsing failed: {e}") from e

    if not isinstance(extra_data, dict):
        raise _bad_upstream_extra_data(
            f"JSON object expected, got {type(extra_data).__name__}"
        )

    headers_to_proxy = extra_data.get("headers_to_proxy")
    if headers_to_proxy is None:
        headers_to_proxy = extra_data.get("HEADERS-TO-PROXY")

    if not headers_to_proxy:
        return {}

    if not isinstance(headers_to_proxy, list):
        raise _bad_upstream_extra_data("headers_to_proxy must be a list")

    result: dict[str, str] = {}
    for header_name in headers_to_proxy:
        if not isinstance(header_name, str):
            raise _bad_upstream_extra_data(
                "headers_to_proxy items must be strings"
            )

        header_value = request_headers.get(header_name)
        if header_value is not None:
            result[header_name] = header_value

    return result
