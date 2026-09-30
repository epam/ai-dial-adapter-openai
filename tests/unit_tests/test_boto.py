from typing import Any

from aidial_adapter_openai.utils.boto import close_client, create_client

_REGION = "us-east-1"
_STS_URL = f"https://sts.{_REGION}.amazonaws.com"


def _open_pools(client: Any) -> int:
    return len(client._endpoint.http_session._manager.pools)


def _open_a_pool(client: Any) -> None:
    # Pools are lazy, so a client only holds one once it has served a request.
    client._endpoint.http_session._manager.connection_from_url(_STS_URL)


def test_clients_share_the_parsed_service_model():
    """The shared Session pools the service models, not the connections."""

    one, two = (create_client("sts", region_name=_REGION) for _ in range(2))

    assert one is not two
    assert one._endpoint.http_session is not two._endpoint.http_session
    assert (
        one.meta.service_model._service_description
        is two.meta.service_model._service_description
    )


async def test_close_client_releases_the_connection_pool():
    client = create_client("sts", region_name=_REGION)
    _open_a_pool(client)
    assert _open_pools(client) == 1

    await close_client(client)

    assert _open_pools(client) == 0
