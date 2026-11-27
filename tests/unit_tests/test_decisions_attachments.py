import base64
import copy
import re
from dataclasses import dataclass

import pytest
import respx
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.decisions.attachments import (
    DECISIONS_ATTACHMENT_RULES,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)

_DIAL_URL = "http://test-dial-url"
_IMAGE_URL = "files/images/img.jpg"
_FORBIDDEN_IMAGE_URL = "files/images/forbidden.jpg"


def _image(url: str) -> dict:
    return {"type": "input_image", "image_url": url}


def _question(instructions: str | list) -> dict:
    return {
        "type": "predicate",
        "name": "damaged",
        "instructions": instructions,
    }


def _request(input_: str | list, questions: list | None = None) -> dict:
    return {
        "model": "gpt-6-luna",
        "input": input_,
        "questions": questions or [_question("Is the item damaged?")],
    }


def _message(content: list, with_type: bool) -> dict:
    request = _request([{"role": "user", "content": content}])
    if with_type:
        request["input"][0]["type"] = "message"
    return request


def _tool_output(type_: str, output: list) -> dict:
    return _request([{"type": type_, "call_id": "call-1", "output": output}])


async def _download(file_storage: FileStorage, request: dict) -> bool:
    return await download_dial_urls_in_request(
        file_storage, request, rules=DECISIONS_ATTACHMENT_RULES
    )


@pytest.fixture
def file_storage():
    return FileStorage.create(dial_url=_DIAL_URL, api_key="test-api-key")


@pytest.fixture(params=[True, False], ids=["with_type", "without_type"])
def with_type(request) -> bool:
    return request.param


@dataclass
class UrlCase:
    url: str
    is_dial: bool

    def check_image(self, part: dict) -> None:
        if self.is_dial:
            encoded = base64.b64encode(b"file-content").decode()
            assert part["image_url"] == f"data:image/jpeg;base64,{encoded}"
        else:
            assert part["image_url"] == self.url


@pytest.fixture(
    params=[
        UrlCase(_IMAGE_URL, True),
        UrlCase("https://example.com/img.jpg", False),
        UrlCase("data:image/png;base64,iVBORw0KGgo=", False),
    ],
    ids=["dial_image", "public_url", "data_url"],
)
def url_case(request) -> UrlCase:
    return request.param


class TestDownloadDialUrlsInDecisionsRequest:
    @pytest.fixture(autouse=True)
    def mock_dial_files_api(self):
        with respx.mock(
            base_url=_DIAL_URL + "/v1",
            assert_all_called=False,
            assert_all_mocked=True,
        ) as router:
            router.get(re.compile(r"images/img\.jpg")).respond(
                text="file-content"
            )
            router.get(re.compile(r"images/forbidden\.jpg")).respond(
                status_code=403, text="Access denied"
            )
            yield router

    @pytest.mark.parametrize(
        "req",
        [
            _request("input string"),
            _request("input string", [_question("instructions string")]),
            _request(
                [{"role": "user", "content": "user message"}],
                [_question([{"type": "input_text", "text": "Is it?"}])],
            ),
        ],
        ids=["string input", "string instructions", "no image parts"],
    )
    async def test_noop(self, file_storage, req: dict):
        request = copy.deepcopy(req)

        assert not await _download(file_storage, request)
        assert request == req

    async def test_message_image(
        self, file_storage, with_type: bool, url_case: UrlCase
    ):
        request = _message([_image(url_case.url)], with_type)

        assert await _download(file_storage, request) == url_case.is_dial
        url_case.check_image(request["input"][0]["content"][0])

    @pytest.mark.parametrize(
        "type_", ["function_call_output", "custom_tool_call_output"]
    )
    async def test_tool_output_image(
        self, file_storage, type_: str, url_case: UrlCase
    ):
        request = _tool_output(type_, [_image(url_case.url)])

        assert await _download(file_storage, request) == url_case.is_dial
        url_case.check_image(request["input"][0]["output"][0])

    async def test_question_instructions_image(
        self, file_storage, url_case: UrlCase
    ):
        request = _request(
            "input string",
            [
                _question("instructions string"),
                _question(
                    [
                        {"type": "input_text", "text": "Is it?"},
                        _image(url_case.url),
                    ]
                ),
            ],
        )

        assert await _download(file_storage, request) == url_case.is_dial
        url_case.check_image(request["questions"][1]["instructions"][1])

    async def test_download_error_propagation(self, file_storage):
        request = _message([_image(_FORBIDDEN_IMAGE_URL)], True)

        with pytest.raises(InvalidRequestError) as exc:
            await _download(file_storage, request)

        assert exc.value.message == (
            f"Failed to download file '{_FORBIDDEN_IMAGE_URL}' (status code 403)"
        )
