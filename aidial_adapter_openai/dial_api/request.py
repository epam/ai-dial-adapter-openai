import functools
from collections.abc import Mapping
from typing import Any, Generic, TypeVar

import fastapi
from aidial_sdk.exceptions import InvalidRequestError, RequestValidationError
from pydantic import BaseModel, ValidationError
from typing_extensions import Protocol

from aidial_adapter_openai.utils.log_config import logger

_T = TypeVar("_T", bound=BaseModel)


def parse_configuration(cls: type[_T], data: Any) -> _T | None:
    if (cf := data.get("custom_fields")) is None:
        return None

    if (conf := cf.get("configuration")) is None:
        return None

    try:
        return cls.model_validate(conf)
    except ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        msg = f"Invalid request. Path: 'custom_field.configuration.{path}', error: {error['msg']}"

        raise RequestValidationError(msg)


def collect_message_text_content(message: dict) -> str:
    text = ""
    if content := message.get("content"):
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text += item["text"]
    return text


def extract_max_prompt_tokens(request: dict) -> int | None:
    if (max_prompt_tokens := request.pop("max_prompt_tokens", None)) is None:
        return None

    if not isinstance(max_prompt_tokens, int):
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is not of type 'integer'",
            param="max_prompt_tokens",
        )

    if max_prompt_tokens < 1:
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is less than the minimum of 1",
            param="max_prompt_tokens",
        )

    return max_prompt_tokens


_R = TypeVar("_R", covariant=True)


class _AzureStyleRequestHandler(Protocol, Generic[_R]):
    async def __call__(
        self, deployment_id: str, request: fastapi.Request
    ) -> _R: ...


def override_deployment_id(
    handler: _AzureStyleRequestHandler[_R],
) -> _AzureStyleRequestHandler[_R]:
    # DIAL Core only applies the models[*].overrideName field to
    # the request body. The deployment id path parameter in
    # an Azure OpenAI endpoint remains unchanged.
    # This decorator fixes this.

    # Preserves the handler's return annotation, which FastAPI
    # turns into the route's response model.
    @functools.wraps(handler)
    async def func(deployment_id: str, request: fastapi.Request) -> _R:
        deployment_id = (
            request.headers.get("X-DIAL-OVERRIDE-NAME") or deployment_id
        )
        return await handler(deployment_id, request)

    return func


def get_upstream_endpoint(request_headers: Mapping[str, str]) -> str:
    name = "X-UPSTREAM-ENDPOINT"
    if (endpoint := request_headers.get(name)) is None:
        raise ValueError(f"{name} header is missing in the request.")

    logger.debug(f"upstream endpoint: {endpoint}")
    return endpoint
