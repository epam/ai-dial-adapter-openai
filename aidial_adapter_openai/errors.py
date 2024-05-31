from typing import Optional

from aidial_sdk import HTTPException as DialException
from aidial_sdk.chat_completion import Response


class UserError(Exception):
    """
    The user errors are aimed to a DIAL chat user.
    So whenever an exceptional situation arises that could be handled by a chat user themselves,
    we should raise a UserError with a `display_message` explaining the error and
    an optional `usage` message to help the user understand how to use the application correctly:

    * `error_message` is what the chat user will be shown as an error message,
    * `usage_message` is reported in a `Usage` dialog stage to educate the chat user.

    A typical example of a user error is validation of supported input data attachments.
    The chat user has full control over the list of attachments, so they can fix the issue themselves.
    """

    error_message: str
    usage_message: Optional[str]

    def __init__(self, error_message: str, usage_message: Optional[str] = None):
        self.error_message = error_message
        self.usage_message = usage_message
        super().__init__(self.error_message)

    async def report_usage(self, response: Response) -> None:
        if self.usage_message is not None:
            with response.create_choice() as choice:
                with choice.create_stage("Usage") as stage:
                    stage.append_content(self.usage_message)
            await response.aflush()

    def to_dial_exception(self) -> DialException:
        return DialException(
            status_code=422,
            type="invalid_request_error",
            message=self.error_message,
            display_message=self.error_message,
            code="invalid_argument",
            param=None,
        )


class ValidationError(Exception):
    """
    The validation errors are aimed to a DIAL API client (e.g. DIAL application developer).
    They report in which way the request to the application is invalid.

    Typically the validation errors are raised when the request not semantically valid but syntactically well-formed.

    For example, an application doesn't support tools/functions feature, but the request contains it.
    It's of no use to report such an error to a chat user, because they can't fix it themselves in the chat.
    But the DIAL application developer who has a finer control over the request can fix the issue by modifying the request.
    """

    message: str
    code: str
    status_code: int
    type: str

    def __init__(
        self,
        message: str,
        code: str = "invalid_argument",
        status_code: int = 422,
        type: str = "invalid_request_error",
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.type = type

        super().__init__(self.message)

    def to_dial_exception(self) -> DialException:
        return DialException(
            status_code=self.status_code,
            type=self.type,
            message=self.message,
            code=self.code,
            param=None,
        )


# The third category of errors is everything else, including standard Python exceptions, like ValueError or KeyError.
# These kind of errors are internal to the DIAL application and thus highlight bugs in the application code itself.
# Neither the chat user nor the DIAL application developer can fix the issue, because there is nothing wrong with the request.
# Thus, such errors are simply reported as internal server errors with HTTP code 500.
