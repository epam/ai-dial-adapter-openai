from typing import TypedDict
from aidial_sdk.exceptions import HTTPException as DialException


def insert_into_template(s: str, transformer_value: str):
    return transformer_value.format(s)


def remove_occurances(s: str, transformer_value: str) -> str:
    return s.replace(transformer_value, "")


PROCESSORS_REGISTRY = {
    "insert_into_template": insert_into_template,
    "remove_occurances": remove_occurances,
}


class StringProcessor(TypedDict):
    name: str
    value: str


def process_string(s: str, processor: StringProcessor) -> str:
    processor_name = processor["name"]
    if processor_name not in PROCESSORS_REGISTRY:
        raise DialException(
            message=(
                "Something is wrong with DeploymentTransformations"
                f"config, there is no such operator {processor_name}"
            ),
            status_code=500,
            type="internal_server_error",
        )
    operator_func = PROCESSORS_REGISTRY[processor_name]
    return operator_func(s=s, transformer_value=processor["value"])
