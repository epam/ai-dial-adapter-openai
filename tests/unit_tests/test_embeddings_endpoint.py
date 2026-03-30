import httpx
import respx


@respx.mock
async def test_embeddings_proxies_extra_headers(test_app: httpx.AsyncClient):
    upstream_endpoint = "http://localhost:5001/openai/v1/embeddings"

    def embeddings_handler(request: httpx.Request):
        assert request.headers.get("x-user-id") == "user-1"
        return httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2],
                        "index": 0,
                    }
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    respx.post(upstream_endpoint).mock(side_effect=embeddings_handler)

    response = await test_app.post(
        "/openai/deployments/text-embedding-3-small/embeddings"
        "?api-version=2023-03-15-preview",
        json={"input": "hello"},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            "X-UPSTREAM-EXTRA-DATA": '{"headers_to_proxy": ["x-user-id"]}',
            "x-user-id": "user-1",
        },
    )

    assert response.status_code == 200
