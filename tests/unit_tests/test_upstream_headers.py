import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


def test_get_upstream_extra_headers_valid_json():
    request_headers = {
        "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id", "x-session-id"]}',
        "x-user-id": "user-1",
        "x-session-id": "session-1",
        "x-other": "not-forwarded",
    }

    result = get_upstream_extra_headers(request_headers)

    assert result == {
        "x-user-id": "user-1",
        "x-session-id": "session-1",
    }


def test_get_upstream_extra_headers_invalid_json():
    request_headers = {
        "X-UPSTREAM-EXTRA-DATA": "{invalid-json",
        "x-user-id": "user-1",
    }

    with pytest.raises(DialException) as exc:
        get_upstream_extra_headers(request_headers)

    assert exc.value.status_code == 502
