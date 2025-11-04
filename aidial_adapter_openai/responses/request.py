from typing import assert_never

from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.responses import (
    ResponseFormatTextConfigParam,
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.shared_params.response_format_json_schema import JSONSchema


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
