import json
from typing import Any, List

from openai.types.chat import ChatCompletionMessageParam

from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.utils.openai import user


def _assert_valid_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError:
        raise AssertionError(f"Not a valid JSON: {text!r}")


def build_response_format(s: TestSuite) -> None:
    messages: List[ChatCompletionMessageParam] = [
        user("extract name and surname from 'John Doe' in json format")
    ]

    if s.supports_response_format_json_object:
        s.test_case(
            name="response_format.json_object",
            messages=messages,
            response_format={"type": "json_object"},
            expected=lambda r: isinstance(
                _assert_valid_json(r.content), (dict, list)
            ),
        )

    if s.supports_response_format_json_schema:
        s.test_case(
            name="response_format.json_schema",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "SchemaName",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "FieldForName": {"type": "string"},
                            "FieldForSurname": {"type": "string"},
                        },
                    },
                },
            },
            expected=lambda r: _assert_valid_json(r.content)
            == {
                "FieldForName": "John",
                "FieldForSurname": "Doe",
            },
        )
