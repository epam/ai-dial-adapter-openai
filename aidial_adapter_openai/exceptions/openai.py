import contextlib

from aidial_sdk.exceptions import HTTPException as DialException
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException,
    ResponseWrapper,
    parse_adapter_exception,
)


def convert_openai_exception(e: Exception) -> AdapterException | None:
    match e:
        case ResponseWrapper():
            return e

        case APIStatusError():
            # Non-streaming errors are reported by `openai` library via this exception
            r = e.response
            httpx_headers = r.headers

            # httpx library (used by openai) automatically sets
            # "Accept-Encoding:gzip,deflate" header in requests to the upstream.
            # Therefore, we may receive from the upstream gzip-encoded
            # response along with "Content-Encoding:gzip" header.
            # We either need to encode the response, or
            # remove the "Content-Encoding" header.
            if "Content-Encoding" in httpx_headers:
                del httpx_headers["Content-Encoding"]

            return parse_adapter_exception(
                status_code=r.status_code,
                headers=httpx_headers,
                content=r.text,
            )

        case APITimeoutError():
            return DialException(
                status_code=504,
                type="timeout",
                message="Request timed out",
                display_message="Request timed out. Please try again later.",
            )

        case APIConnectionError():
            return DialException(
                status_code=502,
                type="connection",
                message="Error communicating with OpenAI",
                display_message="OpenAI server is not responsive. Please try again later.",
            )

        case APIError():
            # Streaming errors are reported by `openai` library via this exception
            status_code: int = 500
            if e.code:
                with contextlib.suppress(Exception):
                    status_code = int(e.code)
                if e.code == "rate_limit_exceeded":
                    status_code = 429

            return parse_adapter_exception(
                status_code=status_code,
                headers={},
                content={"error": e.body or {}},
            )

        case _:
            return None
