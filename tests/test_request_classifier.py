from typing import Any

import pytest

from aidial_adapter_openai.utils.request_classifier import (
    does_request_use_functions_or_tools,
)

new_api_use_classifier_dataset = [
    ({}, False),
    ({"tools": []}, True),
    ({"functions": []}, True),
    ({"tool_choice": []}, True),
    ({"function_call": []}, True),
    ({"messages": [{"role": "tool"}]}, True),
    ({"messages": [{"role": "function"}]}, True),
    ({"messages": [{"function_call": {}}]}, True),
    ({"messages": [{"tool_calls": []}]}, True),
]


@pytest.mark.parametrize("request_body, result", new_api_use_classifier_dataset)
def test_new_api_use_classifier(request_body: Any, result: bool):
    assert does_request_use_functions_or_tools(request_body) == result
