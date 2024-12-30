from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from tests.integration_tests.base import TestSuite, include_deployments
from tests.integration_tests.constants import SAMPLE_DOG_RESOURCE
from tests.utils.openai import user_with_attachment_url, user_with_image_url


@include_deployments(
    [
        ChatCompletionDeploymentType.GPT4O,
        ChatCompletionDeploymentType.GPT4O_MINI,
        ChatCompletionDeploymentType.GPT4_VISION,
    ]
)
def build_vision_common(s: TestSuite) -> None:
    s.test_case(
        name="image_in_content_parts",
        messages=[
            user_with_image_url(
                "What animal is on image? Answer in one word",
                SAMPLE_DOG_RESOURCE,
            ),
        ],
        expected=lambda s: "dog" in s.content.lower(),
        max_tokens=5,
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
        max_tokens=5,
    )
