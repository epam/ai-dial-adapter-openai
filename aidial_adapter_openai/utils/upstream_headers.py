from http import HTTPStatus
from typing import Mapping

from aidial_sdk.exceptions import HTTPException
from pydantic import BaseModel, ValidationError


class UpstreamExtraData(BaseModel):
    headers_to_proxy: list[str] = []


def get_upstream_extra_headers(
    request_headers: Mapping[str, str],
) -> dict[str, str]:
    """Extract upstream headers listed in X-UPSTREAM-EXTRA-DATA."""
    extra_data_header = request_headers.get("X-UPSTREAM-EXTRA-DATA")
    if not extra_data_header:
        return {}

    try:
        extra_data = UpstreamExtraData.model_validate_json(extra_data_header)
    except ValidationError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            type="internal_server_error",
            message=f"Invalid X-UPSTREAM-EXTRA-DATA header: {e}",
        ) from e

    result: dict[str, str] = {}
    for header_name in extra_data.headers_to_proxy:
        header_value = request_headers.get(header_name)
        if header_value is not None:
            result[header_name] = header_value

    return result
