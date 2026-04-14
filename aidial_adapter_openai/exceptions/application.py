import httpx
from aidial_adapter_anthropic.adapter import UserError, ValidationError
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException,
    parse_adapter_exception,
)


def convert_application_errors(e: Exception) -> AdapterException | None:
    match e:
        case httpx.HTTPStatusError():
            r = e.response
            if ret := parse_adapter_exception(
                status_code=r.status_code,
                headers={},
                content=r.text,
            ):
                return ret

        case ValidationError():
            return e.to_dial_exception()

        case UserError():
            return e.to_dial_exception()

        case DialException():
            return e

        case _:
            return None
