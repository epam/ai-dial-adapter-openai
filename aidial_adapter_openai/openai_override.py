import json
from json import JSONDecodeError

import wrapt
from openai.api_requestor import APIRequestor
from openai.openai_response import OpenAIResponse


class OpenAIException(Exception):
    def __init__(self, body, code, resp, headers):
        self.body = body
        self.code = code
        self.resp = resp
        self.headers = headers

        super().__init__(resp)


# Overrided to proxy original errors
def handle_error_response_wrapper(wrapped, self, args, kwargs):
    raise OpenAIException(*args)


# Overrided to proxy original errors
def interpret_response_line_wrapper(wrapped, self: APIRequestor, args, kwargs):
    rbody, rcode, rheaders = args
    stream = kwargs.get("stream", False)

    # HTTP 204 response code does not have any content in the body.
    if rcode == 204:
        return OpenAIResponse(None, rheaders)

    if rcode == 503:
        raise self.handle_error_response(  # overrided
            rbody, rcode, None, rheaders, stream_error=False
        )
    try:
        if "text/plain" in rheaders.get("Content-Type", ""):
            data = rbody
        else:
            data = json.loads(rbody)
    except (JSONDecodeError, UnicodeDecodeError):
        raise self.handle_error_response(  # overrided
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
