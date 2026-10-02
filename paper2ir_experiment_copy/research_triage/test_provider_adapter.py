from __future__ import annotations

import json

import pytest

from research_triage.provider_adapter import (
    OpenAICompatibleProvider,
    ProviderAdapterError,
    ProviderConfig,
    deepseek_provider,
)


def _completion_payload() -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1},
    }


def test_http_adapter_transmits_messages_and_tools_without_mutating_inputs():
    captured: dict = {}
    messages = [{"role": "user", "content": "paper text"}]
    tools = [{"type": "function", "function": {"name": "search_claims", "parameters": {"type": "object"}}}]

    def fake_post(url, body, headers, timeout):
        captured.update(url=url, body=body, headers=headers, timeout=timeout)
        return _completion_payload()

    provider = OpenAICompatibleProvider(
        ProviderConfig(
            model="deepseek-v4.1",
            base_url="https://api.deepseek.com/v1",
            api_key="super-secret-key",
            max_tokens=123,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        ),
        http_post=fake_post,
    )
    result = provider.complete(messages=messages, tools=tools, seed=17)

    assert result == _completion_payload()
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["body"]["messages"] == messages
    assert captured["body"]["tools"] == tools
    assert captured["body"]["seed"] == 17
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["headers"]["Authorization"] == "Bearer super-secret-key"
    # The provider does not mutate caller-owned protocol messages.
    assert messages == [{"role": "user", "content": "paper text"}]


def test_http_adapter_decodes_bytes_and_does_not_log_api_key_on_transport_failure():
    secret = "key-that-must-not-appear"

    def fake_post(*_args):
        raise OSError(secret)

    provider = OpenAICompatibleProvider(
        ProviderConfig(model="deepseek-v4.1", base_url="https://api.deepseek.com/v1", api_key=secret),
        http_post=fake_post,
    )
    with pytest.raises(ProviderAdapterError) as caught:
        provider.complete(messages=[], tools=[])
    assert str(caught.value) == "provider HTTP request failed"
    assert secret not in str(caught.value)
    assert secret not in json.dumps(caught.value.audit)


def test_http_adapter_accepts_json_bytes_response():
    provider = OpenAICompatibleProvider(
        ProviderConfig(model="m", base_url="https://example.invalid/v1", api_key="secret"),
        http_post=lambda *_args: json.dumps(_completion_payload()).encode("utf-8"),
    )
    assert provider.complete(messages=[], tools=[])["choices"][0]["message"]["content"] == "{}"


def test_trisol_exec_adapter_transmits_same_openai_payload_without_http():
    captured: dict = {}

    def fake_exec(service, body, timeout):
        captured.update(service=service, body=body, timeout=timeout)
        return json.dumps(_completion_payload()) + "\nexit_code: 0\n"

    provider = OpenAICompatibleProvider(
        ProviderConfig(model="qwen", transport="trisol_exec", service="research-triage"),
        trisol_exec=fake_exec,
    )
    result = provider.complete(
        messages=[{"role": "assistant", "tool_calls": []}],
        tools=[{"type": "function", "function": {"name": "search_paper"}}],
    )
    assert result == _completion_payload()
    assert captured["service"] == "research-triage"
    assert captured["body"]["messages"][0]["role"] == "assistant"
    assert captured["body"]["tools"][0]["function"]["name"] == "search_paper"


def test_adapter_rejects_malformed_provider_response_without_body_leak():
    provider = OpenAICompatibleProvider(
        ProviderConfig(model="m", base_url="https://example.invalid/v1", api_key="secret"),
        http_post=lambda *_args: "not-json",
    )
    with pytest.raises(ProviderAdapterError, match="not a JSON object") as caught:
        provider.complete(messages=[], tools=[])
    assert caught.value.audit == {"failure_class": "response", "error_type": "invalid_json"}


def test_deepseek_factory_is_explicit_and_does_not_contact_network(monkeypatch):
    monkeypatch.setattr(
        "research_triage.provider_adapter._default_http_post",
        lambda *_args: pytest.fail("network must not be called by construction"),
    )
    provider = deepseek_provider(api_key="secret")
    assert provider.config.base_url == "https://api.deepseek.com/v1"
    assert provider.config.model == "deepseek-v4.1"
    metadata = provider.audit_metadata()
    assert metadata["endpoint_host"] == "api.deepseek.com"
    assert "api_key" not in json.dumps(metadata)
    assert "super-secret" not in json.dumps(metadata)
    assert "secret" not in repr(provider.config)


def test_profile_resolution_is_bounded_before_transport(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("research_triage.model_proposer._load_env", lambda: None)
    with pytest.raises(ProviderAdapterError) as caught:
        OpenAICompatibleProvider.from_model_profile("deepseek_v4_1")
    assert caught.value.audit["failure_class"] == "configuration"
    assert caught.value.audit["missing_env"] == "DEEPSEEK_API_KEY"


def test_adapter_validates_seed_and_reserved_extra_fields():
    with pytest.raises(ValueError, match="seed"):
        OpenAICompatibleProvider(
            ProviderConfig(model="m", base_url="https://example.invalid/v1", api_key="secret")
        ).complete(messages=[], tools=[], seed=-1)
    with pytest.raises(ValueError, match="cannot override messages"):
        OpenAICompatibleProvider(
            ProviderConfig(
                model="m",
                base_url="https://example.invalid/v1",
                api_key="secret",
                extra_body={"messages": []},
            ),
            http_post=lambda *_args: _completion_payload(),
        ).complete(messages=[], tools=[])
