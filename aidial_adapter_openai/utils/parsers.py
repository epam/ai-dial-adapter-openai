import re
from enum import Enum
from json import JSONDecodeError
from typing import Any, Dict, List

from fastapi import Request

from aidial_adapter_openai.utils.exceptions import HTTPException


class ApiType(Enum):
    CHAT_COMPLETION = r"(.+?)/openai/deployments/(.+?)/chat/completions"
    EMBEDDING = r"(.+?)/openai/deployments/(.+?)/embeddings"

    def pattern(self):
        return self.value


def parse_upstream(
    api_endpoint: str,
    api_type: ApiType,
) -> tuple[str, str]:
    match = re.search(api_type.pattern(), api_endpoint)

    if not match:
        raise HTTPException(
            "Invalid upstream endpoint format", 400, "invalid_request_error"
        )

    return match[1], match[2]


async def parse_body(
    request: Request,
) -> Dict[str, Any]:
    try:
        data = await request.json()
    except JSONDecodeError as e:
        raise HTTPException(
            "Your request contained invalid JSON: " + str(e),
            400,
            "invalid_request_error",
        )

    if type(data) != dict:
        raise HTTPException(
            str(data) + " is not of type 'object'", 400, "invalid_request_error"
        )

    return data


def parse_deployment_list(deployments: str) -> List[str]:
    if deployments is None:
        return []

    return list(map(str.strip, deployments.split(",")))
