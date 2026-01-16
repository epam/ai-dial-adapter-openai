import openai

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.integration_tests.constants import (
    IMAGE_RESOURCE,
    UNSUPPORTED_IMAGE_RESOURCE,
)
from tests.utils.openai import (
    ExpectedException,
    user_with_attachment_url,
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

    if s.deployment_type == ChatCompletionDeploymentType.ANTHROPIC_MESSAGES_API:
        # Note: in streaming mode, Anthropic doesn't return error right away,
        # it reports as one of the chunks, therefore, it's not APIStatusError, but
        # rather runtime APIError.
        expected_unsupported = ExpectedException(
            type=openai.APIError,
            message="Unsupported media type: image/bmp",
        )
    else:
        expected_unsupported = ExpectedException(
            type=openai.BadRequestError,
            display_message="The provided image attachment is either corrupt or of unsupported MIME type",
        )

    s.test_case(
        name="unsupported_image_in_attachments",
        messages=[
            user_with_attachment_url(
                "What animal is on image? Answer in one word",
                UNSUPPORTED_IMAGE_RESOURCE,
            ),
        ],
        expected=expected_unsupported,
    )
