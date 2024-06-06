"""
OpenAI SDK translates various HTTP errors received from OpenAI API
into Python exceptions: error.RateLimitError, error.InvalidRequestError,
error.AuthenticationError etc.

We want to retranslate the original HTTP errors to the user.
So the standard error handlers in the openai.api_requestor.APIRequestor class
are rewritten to wrap the original HTTP errors into OpenAIException and raise it.
"""

import asyncio
import json
from json import JSONDecodeError
from typing import AsyncGenerator, Iterator, Tuple, Union

import aiohttp
import wrapt
from openai import error, util
from openai.api_requestor import APIRequestor, parse_stream, parse_stream_async
from openai.openai_response import OpenAIResponse


class OpenAIException(Exception):
    def __init__(self, body, code, resp, headers):
        self.body = body
        self.code = code
        self.resp = resp
        self.headers = headers

        super().__init__(resp)


# Overridden to proxy original errors
def handle_error_response_wrapper(wrapped, self, args, kwargs):
    raise OpenAIException(*args)


def interpret_response_wrapper(
    wrapped,
    self: APIRequestor,
    args,
    kwargs,
    # self, result: requests.Response, stream: bool
) -> Tuple[Union[OpenAIResponse, Iterator[OpenAIResponse]], bool]:
    """Returns the response(s) and a bool indicating whether it is a stream."""
    result, stream = args
    if stream:
        return (
            self._interpret_response_line(
                line, result.status_code, result.headers, stream=True
            )
            for line in parse_stream(result.iter_lines())
        ), True
    else:
        return (
            self._interpret_response_line(
                result.content.decode("utf-8"),
                result.status_code,
                result.headers,
                stream=False,
            ),
            False,
        )


async def interpret_async_response_wrapper(
    wrapped,
    self: APIRequestor,
    args,
    kwargs,
    # self, result: aiohttp.ClientResponse, stream: bool
) -> Tuple[Union[OpenAIResponse, AsyncGenerator[OpenAIResponse, None]], bool]:
    """Returns the response(s) and a bool indicating whether it is a stream."""
    result, stream = args
    if stream:
        return (
            self._interpret_response_line(
                line, result.status, result.headers, stream=True
            )
            async for line in parse_stream_async(result.content)
        ), True
    else:
        try:
            await result.read()
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise error.Timeout("Request timed out") from e
        except aiohttp.ClientError as e:
            util.log_warn(e, body=result.content)
        return (
            self._interpret_response_line(
                (await result.read()).decode("utf-8"),
                result.status,
                result.headers,
                stream=False,
            ),
            False,
        )


# Overridden to proxy original errors
def interpret_response_line_wrapper(wrapped, self: APIRequestor, args, kwargs):
    rbody, rcode, rheaders = args
    stream = kwargs.get("stream", False)
    # HTTP 204 response code does not have any content in the body.
    if rcode == 204:
        return OpenAIResponse(None, rheaders)

    if rcode == 503:
        raise self.handle_error_response(  # overridden
            rbody, rcode, None, rheaders, stream_error=False
        )
    try:
        if "text/plain" in rheaders.get("Content-Type", ""):
            data = rbody
        else:
            data = json.loads(rbody)
    except (JSONDecodeError, UnicodeDecodeError):
        raise self.handle_error_response(  # overridden
            rbody,
            rcode,
            None,
            rheaders,
            stream_error=False,
        )
    resp = OpenAIResponse(data, rheaders)
    # In the future, we might add a "status" parameter to errors
    # to better handle the "error while streaming" case.
    stream_error = stream and "error" in resp.data
    if stream_error or not 200 <= rcode < 300:
        raise self.handle_error_response(
            rbody, rcode, resp.data, rheaders, stream_error=stream_error
        )
    return resp


wrapt.wrap_function_wrapper(
    APIRequestor, "handle_error_response", handle_error_response_wrapper
)

wrapt.wrap_function_wrapper(
    APIRequestor, "_interpret_response_line", interpret_response_line_wrapper
)

wrapt.wrap_function_wrapper(
    APIRequestor,
    "_interpret_async_response",
    interpret_async_response_wrapper,
)

wrapt.wrap_function_wrapper(
    APIRequestor, "_interpret_response", interpret_response_wrapper
)
