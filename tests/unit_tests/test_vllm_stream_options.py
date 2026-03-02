from unittest.mock import AsyncMock

import pytest

import aidial_adapter_openai.chat_completions.gpt as gpt_module
from aidial_adapter_openai.utils.vllm_tokenizer import VllmTokenizer


@pytest.mark.asyncio
async def test_vllm_stream_options_include_usage_injected(monkeypatch):
    """For vLLM streaming calls, the adapter must force stream_options.include_usage=True."""

    captured: dict = {}

    async def fake_call_with_extra_body(fn, request):
        captured["request"] = request
        # Return a non-stream response object to avoid dealing with AsyncStream
        resp = AsyncMock()
        resp.to_dict.return_value = {"id": "x"}
        resp.usage = None
        return resp

    monkeypatch.setattr(
        gpt_module, "call_with_extra_body", fake_call_with_extra_body
    )

    class _DummyProcessor:
        def __init__(self, file_storage=None):
            pass

        async def transform_messages(self, messages):
            # No transformation in this test
            return [
                gpt_module.MultiModalMessage(raw_message=m) for m in messages
            ]

    monkeypatch.setattr(gpt_module, "ResourceProcessor", _DummyProcessor)

    tokenizer = VllmTokenizer(
        model="m",
        upstream_endpoint="http://localhost:17834/v1/chat/completions",
    )

    request = {
        "model": "m",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }

    client = AsyncMock()

    await gpt_module.vllm_chat_completion(
        request=request,
        request_headers={},
        client=client,
        file_storage=None,
        tokenizer=tokenizer,
        eliminate_empty_choices=False,
    )

    assert captured["request"]["stream"] is True
    assert captured["request"]["stream_options"]["include_usage"] is True


@pytest.mark.asyncio
async def test_vllm_stream_options_include_usage_merged(monkeypatch):
    """If stream_options already exists, include_usage must be set/overridden but other fields kept."""

    captured: dict = {}

    async def fake_call_with_extra_body(fn, request):
        captured["request"] = request
        resp = AsyncMock()
        resp.to_dict.return_value = {"id": "x"}
        resp.usage = None
        return resp

    monkeypatch.setattr(
        gpt_module, "call_with_extra_body", fake_call_with_extra_body
    )

    class _DummyProcessor:
        def __init__(self, file_storage=None):
            pass

        async def transform_messages(self, messages):
            return [
                gpt_module.MultiModalMessage(raw_message=m) for m in messages
            ]

    monkeypatch.setattr(gpt_module, "ResourceProcessor", _DummyProcessor)

    tokenizer = VllmTokenizer(
        model="m",
        upstream_endpoint="http://localhost:17834/v1/chat/completions",
    )

    request = {
        "model": "m",
        "stream": True,
        "stream_options": {"foo": "bar", "include_usage": False},
        "messages": [{"role": "user", "content": "hi"}],
    }

    client = AsyncMock()

    await gpt_module.vllm_chat_completion(
        request=request,
        request_headers={},
        client=client,
        file_storage=None,
        tokenizer=tokenizer,
        eliminate_empty_choices=False,
    )

    so = captured["request"]["stream_options"]
    assert so["foo"] == "bar"
    assert so["include_usage"] is True


@pytest.mark.asyncio
async def test_vllm_non_stream_does_not_inject_stream_options(monkeypatch):
    """For non-stream calls, adapter shouldn't force stream_options."""

    captured: dict = {}

    async def fake_call_with_extra_body(fn, request):
        captured["request"] = request
        resp = AsyncMock()
        resp.to_dict.return_value = {"id": "x"}
        resp.usage = None
        return resp

    monkeypatch.setattr(
        gpt_module, "call_with_extra_body", fake_call_with_extra_body
    )

    class _DummyProcessor:
        def __init__(self, file_storage=None):
            pass

        async def transform_messages(self, messages):
            return [
                gpt_module.MultiModalMessage(raw_message=m) for m in messages
            ]

    monkeypatch.setattr(gpt_module, "ResourceProcessor", _DummyProcessor)

    tokenizer = VllmTokenizer(
        model="m",
        upstream_endpoint="http://localhost:17834/v1/chat/completions",
    )

    request = {
        "model": "m",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }

    client = AsyncMock()

    await gpt_module.vllm_chat_completion(
        request=request,
        request_headers={},
        client=client,
        file_storage=None,
        tokenizer=tokenizer,
        eliminate_empty_choices=False,
    )

    assert "stream_options" not in captured["request"]
