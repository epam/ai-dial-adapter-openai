from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.integration_tests.constants import (
    IMAGE_RESOURCE,
    PDF_DOCUMENT_RESOURCE,
)
from tests.utils.openai import (
    ChatCompletionResult,
    user_with_attachment_url,
    user_with_file_content_part,
    user_with_image_content_part,
)


def build_vision_common(s: TestSuite) -> None:
    if not s.supports_vision:
        return

    s.test_case(
        name="image_in_content_parts",
        messages=[
            user_with_image_content_part(
                "What animal is on image? Answer in one word",
                IMAGE_RESOURCE,
            ),
        ],
        expected=lambda s: "dog" in s.content.lower(),
    )

    s.test_case(
        name="image_in_attachments",
        messages=[
            user_with_attachment_url(
                "What animal is on image? Answer in one word",
                IMAGE_RESOURCE,
            ),
        ],
        expected=lambda s: "dog" in s.content.lower(),
    )

    if s.deployment_type == ChatCompletionDeploymentType.RESPONSES_API:
        # Currently only the Responses API supports file input on Azure OpenAI
        query = (
            "Which novel the first page of the attached document quotes from? "
            "Which animal is depicted on the second page?"
        )

        def _check(s: ChatCompletionResult) -> bool:
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
            expected=_check,
        )

        s.test_case(
            name="document_in_attachments",
            messages=[
                user_with_attachment_url(query, PDF_DOCUMENT_RESOURCE),
            ],
            expected=_check,
        )
