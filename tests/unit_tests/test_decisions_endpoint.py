"""
The Decisions API passthrough, against a mocked upstream.
"""

import base64
import gzip
import json
import time
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from aidial_adapter_openai.dial_api.storage import FileStorage

_AZURE_UPSTREAM = "http://test-upstream-hostname/openai/v1/decisions"
_OPENAI_UPSTREAM = "https://api.openai.com/v1/decisions"
_UPSTREAM_KEY = "test-upstream-key"
_PATH = "/openai/v1/decisions"

_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

_DIAL_URL = "http://test-dial-url"
_IMAGE_URL = "files/images/img.jpg"

# Spacing, key order and a field unknown to the adapter: a request
# with no attachment to inline must reach the upstream byte for byte.
_REQUEST_BODY = (
    b'{"questions": [{"type": "predicate", "name": "damaged",'
    b' "instructions": "Is the item damaged?"}],  "model": "gpt-6-luna",'
    b' "input": "The box arrived crushed.", "brand_new_field": 42}'
)

_RESPONSE = {
    "model": "gpt-6-luna",
    "answers": [{"type": "predicate", "name": "damaged", "probability": 1.0}],
    "usage": {
        "input_tokens": 362,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 362,
    },
}


def _headers(
    upstream_endpoint: str | None = _AZURE_UPSTREAM,
    upstream_key: str | None = _UPSTREAM_KEY,
    **extra: str,
) -> dict[str, str]:
    headers = {
        "Api-Key": "test-adapter-api-key",
        "Content-Type": "application/json",
        **extra,
    }
    if upstream_endpoint is not None:
        headers["X-UPSTREAM-ENDPOINT"] = upstream_endpoint
    if upstream_key is not None:
        headers["X-UPSTREAM-KEY"] = upstream_key
    return headers


async def _post(
    test_app: httpx.AsyncClient,
    body: bytes | dict[str, Any] = _REQUEST_BODY,
    *,
    path: str = _PATH,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return await test_app.post(
        path, content=content, headers=headers or _headers()
    )


def _upstream(url: str = _AZURE_UPSTREAM, **kwargs: Any) -> respx.Route:
    if "content" not in kwargs and "json" not in kwargs:
        kwargs["json"] = _RESPONSE
    return respx.post(url).respond(**kwargs)


def _last_request(route: respx.Route) -> httpx.Request:
    return route.calls.last.request


# Routing and passthrough


@respx.mock
async def test_azure_upstream_auth_and_body(test_app: httpx.AsyncClient):
    upstream = _upstream()

    response = await _post(test_app)

    assert response.status_code == 200
    request = _last_request(upstream)
    assert request.method == "POST"
    assert request.url == _AZURE_UPSTREAM
    assert request.headers["authorization"] == f"Bearer {_UPSTREAM_KEY}"
    assert request.content == _REQUEST_BODY


@respx.mock
async def test_openai_upstream_auth_and_body(test_app: httpx.AsyncClient):
    upstream = _upstream(_OPENAI_UPSTREAM)

    response = await _post(test_app, headers=_headers(_OPENAI_UPSTREAM))

    assert response.status_code == 200
    request = _last_request(upstream)
    assert request.method == "POST"
    assert request.url == _OPENAI_UPSTREAM
    assert request.headers["authorization"] == f"Bearer {_UPSTREAM_KEY}"
    assert request.content == _REQUEST_BODY


@respx.mock
async def test_openai_upstream_without_key(test_app: httpx.AsyncClient):
    upstream = _upstream(_OPENAI_UPSTREAM)

    response = await _post(
        test_app, headers=_headers(_OPENAI_UPSTREAM, upstream_key=None)
    )

    assert response.status_code == 401
    assert (
        response.json()["error"]["message"]
        == "X-UPSTREAM-KEY header is missing"
    )
    assert not upstream.called


@respx.mock
async def test_model_passed_through(test_app: httpx.AsyncClient):
    upstream = _upstream()

    await _post(test_app, {"model": "my-azure-deployment", "input": "x"})

    body = json.loads(_last_request(upstream).content)
    assert body["model"] == "my-azure-deployment"


@respx.mock
async def test_headers_to_proxy(test_app: httpx.AsyncClient):
    upstream = _upstream()

    headers = _headers(
        **{
            "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
            "x-user-id": "user-1",
        }
    )
    response = await _post(test_app, headers=headers)

    assert response.status_code == 200
    assert _last_request(upstream).headers.get("x-user-id") == "user-1"


@respx.mock
async def test_api_version_not_forwarded(test_app: httpx.AsyncClient):
    upstream = _upstream()

    response = await _post(test_app, path=f"{_PATH}?api-version=2025-01-01")

    assert response.status_code == 200
    assert _last_request(upstream).url.query == b""


@respx.mock
async def test_body_byte_identical(test_app: httpx.AsyncClient):
    # Undocumented fields, the upstream's own key order and whitespace
    content = (
        b'{ "usage":{"input_tokens":362,"input_tokens_details":'
        b'{"cached_tokens":0,"cache_write_tokens":0,'
        b'"cache_write_tokens_details":{"ttl_30m_tokens":0,'
        b'"ttl_12h_tokens":0}}},\n  "answers" : [],"model":"gpt-6-luna"}'
    )
    _upstream(content=content, headers={"content-type": "application/json"})

    response = await _post(test_app)

    assert response.status_code == 200
    assert response.content == content


@respx.mock
async def test_response_headers(test_app: httpx.AsyncClient):
    content = json.dumps(_RESPONSE).encode()
    respx.post(_AZURE_UPSTREAM).respond(
        status_code=200,
        stream=httpx.ByteStream(gzip.compress(content)),
        headers={
            "x-request-id": "req-123",
            "Content-Type": "application/json",
            "Content-Length": "1",  # plainly false value
            "Content-Encoding": "gzip",
        },
    )

    response = await _post(test_app)

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert "content-encoding" not in response.headers
    assert response.headers["content-length"] == str(len(content))
    assert response.content == content


@respx.mock
async def test_upstream_400_passed_through(test_app: httpx.AsyncClient):
    content = (
        b'{"error":{"message":"Unknown parameter: \'stream\'.",'
        b'"type":"invalid_request_error","param":"stream",'
        b'"code":"unknown_parameter"}}'
    )
    _upstream(status_code=400, content=content)

    response = await _post(test_app, {"model": "gpt-6-luna", "stream": True})

    # The SDK raises on an upstream error, and the adapter's exception
    # handler re-serializes it: the error is kept, its bytes are not.
    assert response.status_code == 400
    assert response.json() == json.loads(content)


@respx.mock
async def test_upstream_404_passed_through(test_app: httpx.AsyncClient):
    content = (
        b'{"error":{"code":"DeploymentNotFound","message":'
        b'"The API deployment for this resource does not exist."}}'
    )
    _upstream(status_code=404, content=content)

    response = await _post(test_app)

    assert response.status_code == 404
    assert response.json() == json.loads(content)


@respx.mock
async def test_missing_upstream_endpoint(test_app: httpx.AsyncClient):
    response = await _post(test_app, headers=_headers(upstream_endpoint=None))

    assert response.status_code == 500


@respx.mock
async def test_non_decisions_upstream(test_app: httpx.AsyncClient):
    upstream_endpoint = "http://test-upstream-hostname/openai/v1/responses"
    upstream = _upstream(upstream_endpoint)

    response = await _post(test_app, headers=_headers(upstream_endpoint))

    assert response.status_code == 502
    assert not upstream.called


_DATED_AZURE_ERROR = (
    "Invalid upstream endpoint format: Only v1 API upstream endpoints"
    " are supported for Decisions API deployments"
)


@respx.mock
async def test_dated_azure_upstream(test_app: httpx.AsyncClient):
    upstream_endpoint = "http://test-upstream-hostname/openai/decisions"
    upstream = _upstream(upstream_endpoint)

    response = await _post(test_app, headers=_headers(upstream_endpoint))

    assert response.status_code == 502
    assert response.json()["error"]["message"] == _DATED_AZURE_ERROR
    assert not upstream.called


@respx.mock
async def test_dated_azure_deployment_upstream(
    test_app: httpx.AsyncClient,
):
    upstream_endpoint = (
        "http://test-upstream-hostname/openai/deployments/luna/decisions"
    )
    upstream = _upstream(upstream_endpoint)

    response = await _post(test_app, headers=_headers(upstream_endpoint))

    assert response.status_code == 502
    assert response.json()["error"]["message"] == _DATED_AZURE_ERROR
    assert not upstream.called


@respx.mock
async def test_non_object_body(test_app: httpx.AsyncClient):
    upstream = _upstream()

    response = await _post(test_app, b'[{"model": "gpt-6-luna"}]')

    assert response.status_code == 400
    assert not upstream.called


@respx.mock
async def test_only_v1_route_exposed(test_app: httpx.AsyncClient):
    _upstream()

    assert (await _post(test_app)).status_code == 200
    assert (await _post(test_app, path="/openai/decisions")).status_code in (
        404,
        405,
    )


# Caching headers


def _usage_response(input_tokens: int) -> dict[str, Any]:
    return {"json": _RESPONSE | {"usage": {"input_tokens": input_tokens}}}


async def _post_caching(
    test_app: httpx.AsyncClient,
    body: dict[str, Any],
    *,
    caching_enabled: bool = True,
) -> httpx.Response:
    # DIAL Core sends the header if the deployment
    # is marked in listing as supporting auto-caching
    extra = {_CACHE_BREAKPOINT_PATH: "whatever"} if caching_enabled else {}
    return await _post(test_app, body, headers=_headers(**extra))


def _assert_no_cache_headers(response: httpx.Response) -> None:
    assert _CACHE_BREAKPOINT_PATH not in response.headers
    assert _CACHE_EXPIRE_AT not in response.headers


def _explicit_request(*input_: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "gpt-6-luna",
        "input": list(input_),
        "prompt_cache_options": {"mode": "explicit"},
    }


def _text_message(text: str, *, breakpoint: bool = False) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "input_text", "text": text}
    if breakpoint:
        part["prompt_cache_breakpoint"] = {}
    return {"role": "user", "content": [part]}


@respx.mock
async def test_cache_headers_string_input(test_app: httpx.AsyncClient):
    _upstream(**_usage_response(1024))

    before = int(time.time())
    response = await _post_caching(test_app, {"input": "..."})
    after = int(time.time())

    assert response.status_code == 200
    assert response.headers[_CACHE_BREAKPOINT_PATH] == "prefix.body.input[0]"
    expire_at = int(response.headers[_CACHE_EXPIRE_AT])
    assert before + 1800 <= expire_at <= after + 1800


@respx.mock
async def test_cache_headers_list_input(test_app: httpx.AsyncClient):
    _upstream(**_usage_response(2716))

    response = await _post_caching(
        test_app, {"input": [_text_message("m0"), _text_message("m1")]}
    )

    assert response.status_code == 200
    assert response.headers[_CACHE_BREAKPOINT_PATH] == "prefix.body.input[1]"


@respx.mock
async def test_no_cache_headers_below_token_threshold(
    test_app: httpx.AsyncClient,
):
    _upstream(**_usage_response(1023))

    response = await _post_caching(test_app, {"input": "..."})

    assert response.status_code == 200
    _assert_no_cache_headers(response)


@respx.mock
async def test_no_cache_headers_without_dial_request_header(
    test_app: httpx.AsyncClient,
):
    _upstream(**_usage_response(5000))

    response = await _post_caching(
        test_app, {"input": "..."}, caching_enabled=False
    )

    assert response.status_code == 200
    _assert_no_cache_headers(response)


@respx.mock
async def test_no_cache_headers_on_upstream_error(test_app: httpx.AsyncClient):
    _upstream(
        status_code=400,
        json={"error": {"message": "bad", "code": "invalid_value"}},
    )

    response = await _post_caching(test_app, {"input": "..."})

    assert response.status_code == 400
    _assert_no_cache_headers(response)


@pytest.mark.parametrize(
    "usage",
    [None, {"input_tokens": "5000"}, {"input_tokens": True}],
    ids=["usage_missing", "tokens_str", "tokens_bool"],
)
@respx.mock
async def test_no_cache_headers_without_valid_usage(
    test_app: httpx.AsyncClient, usage: dict | None
):
    body = {"model": "gpt-6-luna", "answers": []}
    if usage is not None:
        body["usage"] = usage
    content = json.dumps(body).encode()
    _upstream(content=content)

    response = await _post_caching(test_app, {"input": "..."})

    assert response.status_code == 200
    assert response.content == content
    _assert_no_cache_headers(response)


@respx.mock
async def test_no_cache_headers_for_non_json_body(test_app: httpx.AsyncClient):
    content = b"<html>not json</html>"
    _upstream(content=content)

    response = await _post_caching(test_app, {"input": "..."})

    assert response.status_code == 200
    assert response.content == content
    _assert_no_cache_headers(response)


@respx.mock
async def test_no_cache_headers_in_explicit_mode_without_breakpoints(
    test_app: httpx.AsyncClient,
):
    _upstream(**_usage_response(5000))

    response = await _post_caching(
        test_app, _explicit_request(_text_message("m0"), _text_message("m1"))
    )

    assert response.status_code == 200
    _assert_no_cache_headers(response)


@respx.mock
async def test_cache_headers_at_last_explicit_breakpoint(
    test_app: httpx.AsyncClient,
):
    _upstream(**_usage_response(5000))

    response = await _post_caching(
        test_app,
        _explicit_request(
            _text_message("m0", breakpoint=True), _text_message("m1")
        ),
    )

    assert response.status_code == 200
    assert response.headers[_CACHE_BREAKPOINT_PATH] == "prefix.body.input[0]"


# Attachments


@pytest.fixture
def mock_file_storage():
    file_storage = FileStorage.create(
        dial_url=_DIAL_URL, api_key="test-api-key"
    )
    with patch(
        "aidial_adapter_openai.decisions.endpoint.create_file_storage",
        return_value=file_storage,
    ):
        yield


def _image_request(image_url: str) -> bytes:
    return (
        b'{"model": "gpt-6-luna",  "input": [{"role": "user", "content":'
        b' [{"type": "input_image", "image_url": "'
        + image_url.encode()
        + b'"}]}], "questions": []}'
    )


@respx.mock
async def test_dial_image_inlined(
    test_app: httpx.AsyncClient, mock_file_storage
):
    respx.get(f"{_DIAL_URL}/v1/{_IMAGE_URL}").respond(text="file-content")
    upstream = _upstream()

    response = await _post(test_app, _image_request(_IMAGE_URL))

    assert response.status_code == 200
    body = json.loads(_last_request(upstream).content)
    encoded = base64.b64encode(b"file-content").decode()
    assert (
        body["input"][0]["content"][0]["image_url"]
        == f"data:image/jpeg;base64,{encoded}"
    )


@respx.mock
async def test_no_dial_urls_bytes_preserved(
    test_app: httpx.AsyncClient, mock_file_storage
):
    upstream = _upstream()
    request = _image_request("https://example.com/img.jpg")

    response = await _post(test_app, request)

    assert response.status_code == 200
    assert _last_request(upstream).content == request
