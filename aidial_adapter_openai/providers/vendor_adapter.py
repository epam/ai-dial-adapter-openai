from abc import ABC, abstractmethod

from aidial_client.types.chat import ChatCompletionRequest

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)


class VendorAdapter(ABC):
    """
    The deviations of an upstream vendor from the vanilla OpenAI API.
    The vendors which follow the API use the no-op adapter.
    """

    @abstractmethod
    def get_extra_headers(self, deployment_type: D) -> dict[str, str]:
        """
        The headers to add to every upstream request of the deployment.
        """

    @abstractmethod
    def transform_chat_completions_request(
        self, request: ChatCompletionRequest
    ) -> None:
        """
        Rewrites the request in-place as the very last step before it's sent
        to the upstream, when no DIAL-flavoured consumer looks at it anymore.
        """


class NoOpVendorAdapter(VendorAdapter):
    """
    The adapter for the vendors which follow the vanilla OpenAI API.
    """

    def get_extra_headers(self, deployment_type: D) -> dict[str, str]:
        return {}

    def transform_chat_completions_request(
        self, request: ChatCompletionRequest
    ) -> None:
        pass
