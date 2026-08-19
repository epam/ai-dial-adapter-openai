from collections.abc import Mapping
from http import HTTPStatus

from aidial_sdk.exceptions import HTTPException
from pydantic import BaseModel, ValidationError

UPSTREAM_EXTRA_DATA_HEADER = "X-UPSTREAM-EXTRA-DATA"


class UpstreamExtraData(BaseModel):
    """
    The upstream `extra_data` field of the DIAL Core config, delivered to
    the adapter in the `X-UPSTREAM-EXTRA-DATA` header.
    """

    headers_to_proxy: list[str] = []

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_assume_role_arn: str | None = None


def get_upstream_extra_data(
    request_headers: Mapping[str, str],
) -> UpstreamExtraData:
    extra_data_header = request_headers.get(UPSTREAM_EXTRA_DATA_HEADER, "{}")

    try:
        return UpstreamExtraData.model_validate_json(extra_data_header)
    except ValidationError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            type="internal_server_error",
            message=f"Invalid {UPSTREAM_EXTRA_DATA_HEADER} header: {e}",
        ) from e


def get_upstream_extra_headers(
    request_headers: Mapping[str, str],
) -> dict[str, str]:
    extra_data = get_upstream_extra_data(request_headers)

    result: dict[str, str] = {}
    for header_name in extra_data.headers_to_proxy:
        header_value = request_headers.get(header_name)
        if header_value is not None:
            result[header_name] = header_value

    return result
