from dataclasses import dataclass
from functools import cache
from typing import Any, assert_never

import jmespath
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.responses import (
    ResponseFormatTextConfigParam,
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.shared_params.response_format_json_schema import JSONSchema

from aidial_adapter_openai.dial_api.resource import URLResource
from aidial_adapter_openai.dial_api.storage import FileStorage


def convert_response_format(
    response_format: ResponseFormat,
) -> ResponseFormatTextConfigParam:
    match response_format["type"]:
        case "text":
            return response_format
        case "json_object":
            return response_format
        case "json_schema":
            json_schema: JSONSchema = response_format["json_schema"]

            ret = ResponseFormatTextJSONSchemaConfigParam(
                type="json_schema",
                name="json_schema_name",  # invented name since response_format doesn't provide one
                schema=json_schema.get("schema"),  # type: ignore
                strict=json_schema.get("strict"),
            )

            if desc := json_schema.get("description"):
                ret["description"] = desc

            return ret
        case _:
            assert_never(response_format["type"])


async def _download_url_field(
    file_storage: FileStorage, obj: dict, field: str
) -> str | None:
    if not (url := obj.get(field)) or not isinstance(url, str):
        return None

    if not file_storage.is_dial_url(url):
        return None

    dial_resource = URLResource(url=url, entity_name=field)
    resource = await dial_resource.download(file_storage)
    return resource.to_data_url()


@cache
def _compile_jmespath(path: str) -> jmespath.parser.ParsedResult:
    return jmespath.compile(path)


@dataclass
class AttachmentRule:
    path: str
    src_field: str
    dst_field: str | None = None

    async def apply(self, file_storage: FileStorage, request: Any) -> None:
        for match in _compile_jmespath(self.path).search(request):
            obj = match.value
            if not isinstance(obj, dict):
                continue

            if data_url := await _download_url_field(
                file_storage, obj, self.src_field
            ):
                if self.dst_field is None:
                    obj[self.src_field] = data_url
                else:
                    obj.pop(self.src_field, None)
                    obj[self.dst_field] = data_url


_attachment_rules = [
    AttachmentRule(
        "input[?type == null || type == 'message'].content[?type == 'input_image'] | []",
        "image_url",
    ),
    AttachmentRule(
        "input[?type == null || type == 'message'].content[?type == 'input_file'] | []",
        "file_url",
        "file_data",
    ),
    AttachmentRule(
        "input[?type == 'custom_tool_call_output' || type == 'function_call_output'].output[?type == 'input_image'] | []",
        "image_url",
    ),
    AttachmentRule(
        "input[?type == 'custom_tool_call_output' || type == 'function_call_output'].output[?type == 'input_file'] | []",
        "file_url",
        "file_data",
    ),
]


async def download_dial_urls_in_request(
    file_storage: FileStorage | None, request: dict
) -> dict:
    if file_storage is None:
        return request

    for rule in _attachment_rules:
        await rule.apply(file_storage, request)

    return request
