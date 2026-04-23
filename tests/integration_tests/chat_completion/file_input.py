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
    ai_tools,
    tool_request,
    tool_response,
    user,
    user_with_attachment_url,
    user_with_file_content_part,
)


def build_file_input_common(s: TestSuite) -> None:
    if not s.supports_pdf:
        return

    query = (
        "Which novel the first page of the attached document quotes from? "
        "Which animal is depicted on the second page?"
    )

    def expected(s: ChatCompletionResult) -> bool:
        content = s.content.lower()
        success_markers = ["christmas", "carol", "cat"]
        if not any(w in content for w in success_markers):
            assert False, (
                f"Cannot find any of the {success_markers} in the generated content: {content!r}"
            )
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

    if s.deployment_config.type_ == ChatCompletionDeploymentType.RESPONSES_API:
        s.test_case(
            name="document_in_tool_result",
            messages=[
                user(query),
                ai_tools(
                    [tool_request(id="call-id", name="get_document", args={})]
                ),
                tool_response(
                    id="call-id",
                    content="here is the document",
                    resources=[PDF_DOCUMENT_RESOURCE],
                ),
            ],
            expected=expected,
        )

    s.test_case(
        name="unsupported_document_in_attachments",
        messages=[
            user_with_attachment_url(query, UNSUPPORTED_DOCUMENT_RESOURCE),
        ],
        expected=ExpectedException(
            type=(
                openai.BadRequestError,
                openai.UnprocessableEntityError,
                openai.APIError,
            ),
            display_message=r"The file attachments of the MIME type '.*' aren't supported|The provided file attachments aren't supported|Unsupported media type",
        ),
    )
