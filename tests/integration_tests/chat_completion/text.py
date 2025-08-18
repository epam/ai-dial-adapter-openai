from openai import BadRequestError, UnprocessableEntityError

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.utils.openai import ExpectedException, ai, sys, user


def build_text_common(s: TestSuite) -> None:
    if s.supports_reasoning:
        be_brief = {"max_completion_tokens": 512}
    else:
        be_brief = {"max_tokens": 32}

    s.test_case(
        name="dialog recall",
        messages=[
            user("I like Toronto city. Just say hello"),
            ai("Hello"),
            user("what city do I like?"),
        ],
        expected=lambda r: "toronto" in r.content.lower(),
        **be_brief,
    )

    s.test_case(
        name="simple math",
        messages=[user("compute (2+3)")],
        expected=lambda s: "5" in s.content
        and s.response.choices[0].finish_reason == "stop",
    )

    if s.supports_system_prompt:
        s.test_case(
            name="empty system message",
            messages=[sys(""), user("compute (2+4)")],
            expected=lambda s: "6" in s.content,
        )

    if s.deployment_type == ChatCompletionDeploymentType.RESPONSES_API:
        empty_messages_expected = ExpectedException(
            status_code=422,
            type=UnprocessableEntityError,
            message="The request doesn't contain any messages",
        )
    else:
        empty_messages_expected = ExpectedException(
            type=BadRequestError,
            status_code=400,
        )

    s.test_case(
        name="empty dialog",
        messages=[],
        expected=empty_messages_expected,
        **be_brief,
    )

    s.test_case(
        name="empty user message",
        messages=[user("")],
        **be_brief,  # type: ignore
    )

    s.test_case(
        name="single space user message",
        messages=[user(" ")],
        **be_brief,  # type: ignore
    )

    if s.supports_reasoning:
        s.test_case(
            name="short pinocchio",
            messages=[user("tell me the full story of Pinocchio")],
            max_completion_tokens=128,
            reasoning_effort="low",
            expected=lambda s: len(s.response.id) <= 100
            and s.response.choices[0].finish_reason == "length"
            and s.usage is not None,
        )
    else:
        s.test_case(
            name="short pinocchio",
            messages=[user("tell me the full story of Pinocchio")],
            max_tokens=16,
            expected=lambda s: len(s.content.split()) <= 16
            and len(s.response.id) <= 100
            and s.response.choices[0].finish_reason == "length"
            and s.usage is not None
            and s.usage.completion_tokens == 16,
        )

    if s.supports_reasoning_summary:
        s.test_case(
            name="reasoning summary",
            messages=[
                user(
                    "Write a bash script that takes a matrix represented as a string with "
                    'format "[1,2],[3,4],[5,6]" and prints the transpose in the same format.'
                )
            ],
            custom_fields={
                "configuration": {
                    "reasoning": {
                        "effort": "medium",
                        "summary": "auto",
                    }
                }
            },
            expected=lambda s: s.response.choices[0].finish_reason == "stop"
            and len(s.stages) >= 1
            and s.stages[0]["name"] == "Reasoning",
        )

    if s.deployment_type == ChatCompletionDeploymentType.RESPONSES_API:
        multiple_completions_expected = ExpectedException(
            type=UnprocessableEntityError,
            message="The deployment doesn't support request.n parameter other than 1, but got 3.",
            status_code=422,
        )
    else:
        multiple_completions_expected = (
            lambda s: len(s.response.choices) == 3 and s.usage is not None
        )

    s.test_case(
        name="multiple completions",
        n=3,
        messages=[user("2+3=?")],
        expected=multiple_completions_expected,
    )

    if s.supports_temperature:
        s.test_case(
            name="temperature",
            messages=[user("2+3=?")],
            temperature=0.42,
            expected=lambda s: "5" in s.content,
            **be_brief,  # type: ignore
        )


def build_stop_sequence(s: TestSuite) -> None:
    if not s.supports_stop:
        return

    if s.deployment_type == ChatCompletionDeploymentType.RESPONSES_API:
        expected = ExpectedException(
            type=UnprocessableEntityError,
            message="The deployment doesn't support stop request parameter.",
            status_code=422,
        )
    else:
        expected = lambda s: "john" not in s.content.lower()  # noqa: E731

    s.test_case(
        name="stop sequence",
        stop=["John", "john"],
        messages=[user('Reply with "Hello John Doe"')],
        expected=expected,
    )


def build_multi_system(s: TestSuite) -> None:
    if not s.supports_system_prompt:
        return

    messages = [
        sys("act as a helpful assistant"),
        sys("act as a calculator"),
        user("2+5=?"),
    ]

    if s.deployment_type == ChatCompletionDeploymentType.DATABRICKS:
        s.test_case(
            name="many system",
            messages=messages,
            # Databricks does not allow multiple system messages
            expected=ExpectedException(
                type=BadRequestError,
                message="Chat message input roles must alternate",
                status_code=400,
            ),
        )
    else:
        s.test_case(
            name="many system",
            messages=messages,
            expected=lambda s: "7" in s.content.lower(),
        )
