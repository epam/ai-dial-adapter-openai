import json
from copy import deepcopy

import httpx


def _merge_defaults_inplace(dst: dict, defaults: dict) -> dict:
    """Recursively apply defaults into dst (without overriding existing keys)."""
    for k, v in defaults.items():
        if k not in dst:
            dst[k] = deepcopy(v)
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            _merge_defaults_inplace(dst[k], v)
    return dst


def with_request_overrides(
    client: httpx.AsyncClient, defaults: dict | None
) -> httpx.AsyncClient:
    if defaults is None:
        return client

    async def on_request(request: httpx.Request):
        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type:
            return

        if request.content is None:
            return

        try:
            body = json.loads(request.content)
        except Exception:
            return

        if not isinstance(body, dict):
            return

        merged = _merge_defaults_inplace(body, defaults)

        request._content = json.dumps(merged).encode("utf-8")
        request.headers["content-length"] = str(len(request.content))

    new_hooks = dict(client.event_hooks)
    new_hooks.setdefault("request", []).append(on_request)

    return httpx.AsyncClient(
        transport=client._transport,
        headers=client.headers,
        timeout=client.timeout,
        event_hooks=new_hooks,
        base_url=client.base_url,
    )
