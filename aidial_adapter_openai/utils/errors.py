from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.utils.errors import json_error


def dial_exception_to_json_error(exc: DialException) -> dict:
    return json_error(
        message=exc.message,
        type=exc.type,
        param=exc.param,
        code=exc.code,
        display_message=exc.display_message,
    )
