import httpx
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client


def _auth_headers(
    creds: OpenAICreds, headers: dict[str, str] | None
) -> dict[str, str]:
    result = dict(headers or {})
    if token := creds.get("api_key") or creds.get("azure_ad_token"):
        result["Authorization"] = f"Bearer {token}"
    return result


def _raise_on_error(response: httpx.Response) -> None:
    if response.is_success:
        return

    try:
        body = response.json()
    except Exception:
        body = {}

    error = body.get("error") or {}
    message = error.get("message") or response.reason_phrase or "Unknown Error"
    raise DialException(
        message=message,
        status_code=response.status_code,
        type=error.get("type"),
        param=error.get("param"),
        code=error.get("code"),
    )


async def post_upstream(
    *,
    endpoint: str,
    body: dict,
    creds: OpenAICreds,
    headers: dict[str, str] | None,
) -> dict:
    response = await get_http_client().post(
        url=endpoint,
        json=body,
        headers=_auth_headers(creds, headers),
    )
    _raise_on_error(response)
    return response.json()
