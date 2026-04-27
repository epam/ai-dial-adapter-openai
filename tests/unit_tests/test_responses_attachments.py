import base64
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import respx
from openai.types.responses import (
    ResponseCustomToolCallOutputParam,
    ResponseInputFileParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
)
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)
from openai.types.responses.response_custom_tool_call_output_param import (
    OutputOutputContentList,
)
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_input_param import (
    FunctionCallOutput,
    Message,
)

from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)

_API_KEY = "test-dial-api-key"
_DIAL_URL = "http://test-dial-url"
_IMAGE_URL = f"{_DIAL_URL}/images/img.jpg"
_FILE_URL = f"{_DIAL_URL}/documents/doc.pdf"
_GLOBAL_URL = "http://example.com/file"
Request = dict[str, Any]


def _data_url(content_type: str, url: str) -> str:
    content = f"file-content:{url}".encode()
    encoded = base64.b64encode(content).decode()
    return f"data:{content_type};base64,{encoded}"


def _message_image(url: str) -> ResponseInputImageParam:
    return ResponseInputImageParam(
        type="input_image",
        image_url=url,
        detail="auto",
    )


def _message_file(url: str) -> ResponseInputFileParam:
    return ResponseInputFileParam(type="input_file", file_url=url)


def _function_output_image(url: str) -> ResponseInputImageContentParam:
    return ResponseInputImageContentParam(
        type="input_image",
        image_url=url,
        detail="auto",
    )


def _tool_output_file(url: str) -> ResponseInputFileParam:
    return ResponseInputFileParam(type="input_file", file_url=url)


def _message_request(content: ResponseInputMessageContentListParam) -> Request:
    request = ResponseCreateParamsBase(
        model="test-model",
        input=[Message(role="user", content=content)],
    )
    return dict(request)


def _function_output_request(
    output: ResponseFunctionCallOutputItemListParam,
) -> Request:
    request = ResponseCreateParamsBase(
        model="test-model",
        input=[
            FunctionCallOutput(
                type="function_call_output",
                call_id="call-1",
                output=output,
            )
        ],
    )
    return dict(request)


def _custom_output_request(output: list[OutputOutputContentList]) -> Request:
    request = ResponseCreateParamsBase(
        model="test-model",
        input=[
            ResponseCustomToolCallOutputParam(
                type="custom_tool_call_output",
                call_id="call-1",
                output=output,
            )
        ],
    )
    return dict(request)


@dataclass(frozen=True)
class Case:
    name: str
    request_factory: Callable[[], Request]
    part: Callable[[Request], dict[str, Any]]
    src_field: str
    dst_field: str
    expected: str
    invalid_value: str | dict[str, str]


CASES = (
    Case(
        name="message-image",
        request_factory=lambda: _message_request([_message_image(_IMAGE_URL)]),
        part=lambda request: request["input"][0]["content"][0],
        src_field="image_url",
        dst_field="image_url",
        expected=_data_url("image/jpeg", _IMAGE_URL),
        invalid_value=_GLOBAL_URL,
    ),
    Case(
        name="message-file",
        request_factory=lambda: _message_request([_message_file(_FILE_URL)]),
        part=lambda request: request["input"][0]["content"][0],
        src_field="file_url",
        dst_field="file_data",
        expected=_data_url("application/pdf", _FILE_URL),
        invalid_value={"url": _FILE_URL},
    ),
    Case(
        name="function-output-image",
        request_factory=lambda: _function_output_request(
            [_function_output_image(_IMAGE_URL)]
        ),
        part=lambda request: request["input"][0]["output"][0],
        src_field="image_url",
        dst_field="image_url",
        expected=_data_url("image/jpeg", _IMAGE_URL),
        invalid_value=_GLOBAL_URL,
    ),
    Case(
        name="custom-output-file",
        request_factory=lambda: _custom_output_request(
            [_tool_output_file(_FILE_URL)]
        ),
        part=lambda request: request["input"][0]["output"][0],
        src_field="file_url",
        dst_field="file_data",
        expected=_data_url("application/pdf", _FILE_URL),
        invalid_value={"url": _FILE_URL},
    ),
)


@pytest.fixture
def dial_url_env(monkeypatch):
    monkeypatch.setenv("DIAL_URL", _DIAL_URL)
    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.DIAL_URL", _DIAL_URL
    )
    monkeypatch.setattr(
        "aidial_adapter_openai.dial_api.storage.DIAL_USE_FILE_STORAGE", True
    )


@pytest.fixture
def file_storage(dial_url_env):
    storage = create_file_storage({"api-key": _API_KEY})
    assert storage is not None
    return storage


@pytest.fixture
def mock_dial_files_api():
    pattern = re.compile(
        rf"{re.escape(_DIAL_URL)}/(images/img\.jpg|documents/doc\.pdf)"
    )

    with respx.mock(assert_all_called=False) as router:
        route = router.get(pattern).mock(
            side_effect=lambda request: httpx.Response(
                200, content=f"file-content:{request.url}"
            )
        )
        yield route


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_download_dial_urls_in_request_positive(
    file_storage,
    mock_dial_files_api,
    case: Case,
):
    request = case.request_factory()

    await download_dial_urls_in_request(file_storage, request)

    part = case.part(request)
    assert part[case.dst_field] == case.expected
    assert (case.src_field in part) is (case.src_field == case.dst_field)
    assert mock_dial_files_api.call_count == 1


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_download_dial_urls_in_request_negative(
    file_storage,
    mock_dial_files_api,
    case: Case,
):
    request = case.request_factory()
    part = case.part(request)
    part[case.src_field] = case.invalid_value

    await download_dial_urls_in_request(file_storage, request)

    assert part[case.src_field] == case.invalid_value
    assert case.dst_field not in part or case.dst_field == case.src_field
    assert mock_dial_files_api.call_count == 0
