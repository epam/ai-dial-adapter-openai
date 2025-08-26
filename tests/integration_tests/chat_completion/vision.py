import openai

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

    s.test_case(
        name="unsupported_image_in_attachments",
        messages=[
            user_with_attachment_url(
                "What animal is on image? Answer in one word",
                UNSUPPORTED_IMAGE_RESOURCE,
            ),
        ],
        expected=ExpectedException(
            type=openai.BadRequestError,
            display_message="The provided image attachment is either corrupt or of unsupported MIME type",
        ),
    )
