import anthropic
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client

_CORE = "http://dial-core"


@respx.mock
async def test_2_plus_3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIAL_URL", _CORE)
    route = respx.post(f"{_CORE}/openai/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "5"},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    async with (
        create_test_client(ApplicationConfig()) as test_app,
        anthropic.AsyncAnthropic(
            api_key="dummy-key",
            base_url=f"{test_app.base_url}/openaiChatCompletions/anthropic",
            http_client=test_app,
            max_retries=0,
        ) as client,
    ):
        response = await client.messages.create(
            model="gpt-5.5",
            max_tokens=100,
            messages=[{"role": "user", "content": "2+3=?"}],
        )

    assert route.call_count == 1
    assert response.content[0].type == "text"
    assert response.content[0].text == "5"
