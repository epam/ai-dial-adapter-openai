import openai

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.integration_tests.constants import (
    PDF_DOCUMENT_RESOURCE,
    UNSUPPORTED_DOCUMENT_RESOURCE,
)
from tests.utils.openai import (
    ChatCompletionResult,
    ExpectedException,
    user_with_attachment_url,
    user_with_file_content_part,
)


def build_file_input_common(s: TestSuite) -> None:
    if (
        not s.supports_vision
        or s.deployment_type != ChatCompletionDeploymentType.RESPONSES_API
    ):
        return

    query = (
        "Which novel the first page of the attached document quotes from? "
        "Which animal is depicted on the second page?"
    )

    def expected(s: ChatCompletionResult) -> bool:
        content = s.content.lower()
        for w in ["christmas", "carol", "cat"]:
            assert w in content
        return True

    s.test_case(
        name="document_in_content_parts",
        messages=[
            user_with_file_content_part(
                query, "document.pdf", PDF_DOCUMENT_RESOURCE
            ),
        ],
        expected=expected,
    )

    s.test_case(
        name="document_in_attachments",
        messages=[
            user_with_attachment_url(query, PDF_DOCUMENT_RESOURCE),
        ],
        expected=expected,
    )

    s.test_case(
        name="unsupported_document_in_attachments",
        messages=[
            user_with_attachment_url(query, UNSUPPORTED_DOCUMENT_RESOURCE),
        ],
        expected=ExpectedException(
            type=openai.BadRequestError,
            display_message=r"The file attachments of the MIME type '.*' aren't supported|The provided file attachments aren't supported",
        ),
    )
