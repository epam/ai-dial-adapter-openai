from openai import BadRequestError, UnprocessableEntityError

from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from tests.integration_tests.base import TestSuite
from tests.utils.openai import ExpectedException, ai, sys, user


def build_text_common(s: TestSuite) -> None:
    # Basic dialog tests
    s.test_case(
        name="dialog recall",
        messages=[
            user("I like Toronto city. Just say hello"),
            ai("Hello"),
            user("what city do I like?"),
        ],
        max_tokens=10,
        expected=lambda r: "toronto" in r.content.lower(),
    )

    s.test_case(
        name="simple math",
        messages=[user("compute (2+3)")],
        expected=lambda s: "5" in s.content,
    )

    s.test_case(
        name="empty system message",
        messages=[sys(""), user("compute (2+4)")],
        expected=lambda s: "6" in s.content,
    )

    if s.deployment_type == ChatCompletionDeploymentType.GPT_TEXT_ONLY:
        expected_exc = ExpectedException(
            type=BadRequestError,
            message="should be non-empty",
            status_code=400,
        )
    elif s.deployment_type == ChatCompletionDeploymentType.MISTRAL:
        expected_exc = ExpectedException(
            type=BadRequestError,
            message="Conversation must have at least one message",
            status_code=400,
        )
    elif s.deployment_type == ChatCompletionDeploymentType.DATABRICKS:
        expected_exc = ExpectedException(
            type=BadRequestError,
            message="cannot be an empty list",
            status_code=400,
        )
    else:
        expected_exc = ExpectedException(
            type=UnprocessableEntityError,
            message="The request doesn't contain any messages",
            status_code=422,
        )
    s.test_case(
        name="empty dialog",
        max_tokens=1,
        messages=[],
        expected=expected_exc,
    )

    s.test_case(
        name="empty user message",
        max_tokens=1,
        messages=[user("")],
    )
    s.test_case(
        name="single space user message",
        max_tokens=1,
        messages=[user(" ")],
    )

    s.test_case(
        name="pinocchio in one token",
        max_tokens=1,
        messages=[user("tell me the full story of Pinocchio")],
        expected=lambda s: len(s.content.split()) <= 1,
    )


def build_stop_sequence(s: TestSuite) -> None:
    if s.deployment_type == ChatCompletionDeploymentType.MISTRAL:
        # Mistral just ignores stop sequence

        def expected(s) -> bool:
            return "john" in s.content.lower()

    else:

        def expected(s) -> bool:
            return "john" not in s.content.lower()

    s.test_case(
        name="stop sequence",
        stop=["John", "john"],
        messages=[user('Reply with "Hello John Doe"')],
        expected=expected,
    )


def build_multi_system(s: TestSuite) -> None:
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
                message=("Chat message input roles must alternate"),
                status_code=400,
            ),
        )
    else:
        s.test_case(
            name="many system",
            messages=messages,
            expected=lambda s: "7" in s.content.lower(),
        )
