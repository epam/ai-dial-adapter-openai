import httpx

from aidial_adapter_openai.constant import (
    DEFAULT_CONNECTION_LIMITS,
    DEFAULT_TIMEOUT,
)

http_client = httpx.AsyncClient(
    timeout=DEFAULT_TIMEOUT,
    limits=DEFAULT_CONNECTION_LIMITS,
    follow_redirects=True,
)
