from tests.integration_tests.chat_completion.test_case import TestSuite
from tests.integration_tests.constants import SAMPLE_DOG_RESOURCE
from tests.utils.openai import user_with_attachment_url, user_with_image_url


def build_vision_common(s: TestSuite) -> None:
    if not s.supports_vision:
        return

    s.test_case(
        name="image_in_content_parts",
        messages=[
            user_with_image_url(
                "What animal is on image? Answer in one word",
                SAMPLE_DOG_RESOURCE,
            ),
        ],
        expected=lambda s: "dog" in s.content.lower(),
        max_tokens=16,
    )

    s.test_case(
        name="image_in_custom_content",
        messages=[
            user_with_attachment_url(
                "What animal is on image? Answer in one word",
                SAMPLE_DOG_RESOURCE,
            ),
        ],
        expected=lambda s: "dog" in s.content.lower(),
        max_tokens=16,
    )
