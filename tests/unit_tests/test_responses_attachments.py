import base64
import re
from dataclasses import dataclass

import pytest
import respx
from openai.types.responses import (
    ResponseCustomToolCallOutputParam,
    ResponseInputFileContentParam,
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
from pydantic import SecretStr

from aidial_adapter_openai.dial_api.storage import (
    FileStorage,
)
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)

_DIAL_URL = "http://test-dial-url"
_IMAGE_URL = "images/img.jpg"
_FILE_URL = "documents/doc.pdf"


def _image(url: str) -> ResponseInputImageParam:
    return ResponseInputImageParam(
        type="input_image", image_url=url, detail="auto"
    )


def _image_content(url: str) -> ResponseInputImageContentParam:
    return ResponseInputImageContentParam(type="input_image", image_url=url)


def _file(url: str) -> ResponseInputFileParam:
    return ResponseInputFileParam(type="input_file", file_url=url)


def _file_content(url: str) -> ResponseInputFileContentParam:
    return ResponseInputFileContentParam(type="input_file", file_url=url)


def _message(
    content: ResponseInputMessageContentListParam, with_type: bool
) -> ResponseCreateParamsBase:
    message = Message(role="user", content=content)
    if with_type:
        message["type"] = "message"
    return ResponseCreateParamsBase(model="test-model", input=[message])


def _function_call(
    output: ResponseFunctionCallOutputItemListParam,
) -> ResponseCreateParamsBase:
    request = ResponseCreateParamsBase(
        model="test-model",
        input=[
            FunctionCallOutput(
                type="function_call_output", call_id="call-1", output=output
            )
        ],
    )
    return request


def _custom_tool_call(
    output: list[OutputOutputContentList],
) -> ResponseCreateParamsBase:
    request = ResponseCreateParamsBase(
        model="test-model",
        input=[
            ResponseCustomToolCallOutputParam(
                type="custom_tool_call_output", call_id="call-1", output=output
            )
        ],
    )
    return request


@pytest.fixture
def file_storage():
    return FileStorage(dial_url=_DIAL_URL, api_key=SecretStr("test-api-key"))


@pytest.fixture(params=[True, False], ids=["with_type", "without_type"])
def with_type(request) -> bool:
    return request.param


@pytest.fixture(autouse=True)
def mock_dial_files_api():
    pattern = re.compile(r"(images/img\.jpg|documents/doc\.pdf)")
    base_url = _DIAL_URL + "/v1"
    with respx.mock(
        base_url=base_url,
        assert_all_called=False,
        assert_all_mocked=True,
    ) as router:
        yield router.get(pattern).respond(text="file-content")


@dataclass
class UrlCase:
    url: str
    type_: str
    is_dial: bool

    @property
    def expected_data_url(self) -> str:
        encoded = base64.b64encode(b"file-content").decode()
        return f"data:{self.type_};base64,{encoded}"

    def check_image(self, part: dict) -> None:
        if self.is_dial:
            assert part["image_url"] == self.expected_data_url
        else:
            assert part["image_url"] == self.url

    def check_file(self, part: dict) -> None:
        if self.is_dial:
            assert part["file_data"] == self.expected_data_url
            assert "file_url" not in part
        else:
            assert part["file_url"] == self.url
            assert "file_data" not in part


@pytest.fixture(
    params=[
        UrlCase(_IMAGE_URL, "image/jpeg", True),
        UrlCase(_FILE_URL, "application/pdf", True),
        UrlCase("http://example.com/file.txt", "text/plain", False),
    ],
    ids=["dial_image", "dial_doc", "external_url"],
)
def url_case(request) -> UrlCase:
    return request.param


async def test_download_dial_urls_in_request_message_image(
    file_storage, with_type: bool, url_case: UrlCase
):
    request = _message([_image(url_case.url)], with_type)

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["content"][0]  # type: ignore

    url_case.check_image(part)


async def test_download_dial_urls_in_request_message_file(
    file_storage, with_type: bool, url_case: UrlCase
):
    request = _message([_file(url_case.url)], with_type)

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["content"][0]  # type: ignore

    url_case.check_file(part)


async def test_download_dial_urls_in_request_function_output_image(
    file_storage, url_case: UrlCase
):
    request = _function_call([_image_content(url_case.url)])

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["output"][0]  # type: ignore

    url_case.check_image(part)


async def test_download_dial_urls_in_request_function_output_file(
    file_storage, url_case: UrlCase
):
    request = _function_call([_file_content(url_case.url)])

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["output"][0]  # type: ignore

    url_case.check_file(part)


async def test_download_dial_urls_in_request_custom_output_image(
    file_storage, url_case: UrlCase
):
    request = _custom_tool_call([_image(url_case.url)])

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["output"][0]  # type: ignore

    url_case.check_image(part)


async def test_download_dial_urls_in_request_custom_output_file(
    file_storage, url_case: UrlCase
):
    request = _custom_tool_call([_file(url_case.url)])

    result = await download_dial_urls_in_request(file_storage, request)
    part: dict = result["input"][0]["output"][0]  # type: ignore

    url_case.check_file(part)
