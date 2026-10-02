"""OpenAI-compatible completion providers for the v2 tool loop.

The v2 runner owns tool execution and evidence accounting.  This module only
transports a ``messages``/``tools`` request to an OpenAI-compatible endpoint
and returns the decoded response object.  It deliberately keeps credentials
out of request metadata and exception messages.  Both the public DeepSeek
endpoint and the authenticated Trisol pod-local ``exec`` path use the same
``CompletionProvider.complete`` contract.

Tests inject ``http_post`` or ``trisol_exec`` callables, so the test suite does
not open a socket or invoke the Trisol CLI.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class ProviderAdapterError(RuntimeError):
    """A bounded provider transport or response-protocol failure.

    ``audit`` contains only non-secret labels, sizes, and hashes where
    applicable.  It never contains the API key, request body, or response
    content.
    """

    def __init__(self, message: str, audit: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.audit = dict(audit or {})


HttpPost = Callable[[str, dict[str, Any], dict[str, str], float], Any]
TrisolExec = Callable[[str, dict[str, Any], float], Any]


def _decode_payload(raw: Any, *, source: str) -> dict[str, Any]:
    """Decode a dict, JSON bytes/string, or urllib-like response object."""

    value = raw
    if hasattr(value, "read") and callable(value.read):
        value = value.read()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProviderAdapterError(
                f"{source} response is not UTF-8 JSON",
                {"failure_class": "response", "error_type": "invalid_utf8"},
            ) from exc
    if isinstance(value, str):
        value = value.strip()
        # ``trisol inference exec`` appends this wrapper line after stdout.
        value = re.sub(r"\nexit_code:\s*0\s*$", "", value).strip()
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProviderAdapterError(
                f"{source} response is not a JSON object",
                {"failure_class": "response", "error_type": "invalid_json"},
            ) from exc
    if not isinstance(value, dict):
        raise ProviderAdapterError(
            f"{source} response is not a JSON object",
            {"failure_class": "response", "error_type": "invalid_payload"},
        )
    return copy.deepcopy(value)


def _default_http_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    """POST JSON over HTTPS without logging request or credential material."""

    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        # Do not stringify the exception: servers occasionally echo request
        # headers or other sensitive data in their error text.
        raise ProviderAdapterError(
            "provider HTTP request failed",
            {"failure_class": "transport", "error_type": "http", "status": exc.code},
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderAdapterError(
            "provider HTTP request failed",
            {"failure_class": "transport", "error_type": type(exc).__name__},
        ) from exc


def _default_trisol_exec(service: str, body: dict[str, Any], timeout: float) -> Any:
    """Run a request through the already-authenticated Trisol service pod."""

    if not isinstance(service, str) or not service.strip():
        raise ProviderAdapterError(
            "Trisol service is not configured",
            {"failure_class": "configuration", "missing": "service"},
        )
    encoded = base64.b64encode(json.dumps(body, ensure_ascii=False).encode("utf-8")).decode("ascii")
    script = (
        "import base64,json,sys,urllib.request; "
        "p=json.loads(base64.b64decode(sys.argv[1])); "
        "r=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions', "
        "data=json.dumps(p,ensure_ascii=False).encode('utf-8'), "
        "headers={'Content-Type':'application/json'},method='POST'); "
        "print(urllib.request.urlopen(r,timeout=300).read().decode('utf-8'))"
    )
    try:
        completed = subprocess.run(
            ["trisol", "inference", "exec", service, "--", "python3", "-c", script, encoded],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderAdapterError(
            "Trisol exec request failed",
            {"failure_class": "transport", "error_type": type(exc).__name__},
        ) from exc
    if completed.returncode != 0:
        raise ProviderAdapterError(
            "Trisol exec request failed",
            {"failure_class": "transport", "error_type": "nonzero_exit", "exit_code": completed.returncode},
        )
    return completed.stdout


@dataclass(frozen=True)
class ProviderConfig:
    """Non-secret transport settings for one model endpoint."""

    model: str
    base_url: str | None = None
    # ``repr=False`` prevents accidental provider/config logging from copying
    # a credential into a run artifact or terminal transcript.
    api_key: str | None = field(default=None, repr=False)
    transport: str = "http"
    service: str | None = None
    timeout: float = 240.0
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None


class OpenAICompatibleProvider:
    """A ``CompletionProvider`` backed by HTTP or Trisol pod-local exec."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        http_post: HttpPost | None = None,
        trisol_exec: TrisolExec | None = None,
    ) -> None:
        if not isinstance(config, ProviderConfig):
            raise TypeError("config must be a ProviderConfig")
        if not isinstance(config.model, str) or not config.model.strip():
            raise ValueError("model must be a non-empty string")
        if config.transport not in {"http", "trisol_exec"}:
            raise ValueError("transport must be 'http' or 'trisol_exec'")
        if config.transport == "http":
            if not isinstance(config.base_url, str) or not config.base_url.strip():
                raise ValueError("base_url is required for HTTP transport")
            if not isinstance(config.api_key, str) or not config.api_key:
                raise ValueError("api_key is required for HTTP transport")
        elif not isinstance(config.service, str) or not config.service.strip():
            raise ValueError("service is required for Trisol transport")
        if isinstance(config.timeout, bool) or not isinstance(config.timeout, (int, float)) or config.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.config = config
        self._http_post = http_post or _default_http_post
        self._trisol_exec = trisol_exec or _default_trisol_exec

    @classmethod
    def from_model_profile(cls, profile_name: str = "deepseek_v4_1") -> "OpenAICompatibleProvider":
        """Resolve the repository's existing profile without exposing secrets.

        Profile loading is deliberately lazy, so importing this adapter never
        reads ``.env`` or contacts a provider.  The existing model profile
        definitions remain the single source of endpoint/model defaults.
        """

        from .model_proposer import _load_env, _resolve_model_profile

        _load_env()
        _, profile = _resolve_model_profile(profile_name)
        api_key_env = profile["api_key_env"]
        base_url_env = profile["base_url_env"]
        model_env = profile["model_env"]
        service_env = profile.get("service_env")
        api_key = os.environ.get(api_key_env)
        base_url = os.environ.get(base_url_env, profile.get("default_base_url"))
        model = os.environ.get(model_env, profile.get("default_model"))
        service = os.environ.get(service_env, profile.get("default_service")) if isinstance(service_env, str) else None
        transport = profile.get("transport", "http")
        if profile.get("provider") == "trisol" and isinstance(service_env, str) and os.environ.get(service_env):
            transport = "trisol_exec"
        if transport == "trisol_exec":
            if not isinstance(service, str) or not service.strip():
                raise ProviderAdapterError(
                    "model configuration is incomplete",
                    {"failure_class": "configuration", "missing_env": service_env},
                )
            return cls(
                ProviderConfig(
                    model=model,
                    transport="trisol_exec",
                    service=service,
                    timeout=240.0,
                    max_tokens=profile.get("max_tokens"),
                    response_format={"type": "json_object"},
                )
            )
        if not isinstance(api_key, str) or not api_key:
            raise ProviderAdapterError(
                "model configuration is incomplete",
                {"failure_class": "configuration", "missing_env": api_key_env},
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise ProviderAdapterError(
                "model configuration is incomplete",
                {"failure_class": "configuration", "missing_env": base_url_env},
            )
        return cls(
            ProviderConfig(
                model=model,
                base_url=base_url,
                api_key=api_key,
                transport="http",
                timeout=240.0,
                max_tokens=profile.get("max_tokens"),
                response_format={"type": "json_object"},
                extra_body=(
                    {"thinking": {"type": "disabled"}}
                    if profile.get("provider") == "deepseek"
                    else None
                ),
            )
        )

    def _request_body(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], seed: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "temperature": 0,
        }
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if self.config.response_format is not None:
            body["response_format"] = copy.deepcopy(self.config.response_format)
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer or None")
            body["seed"] = seed
        for key, value in (self.config.extra_body or {}).items():
            if key in {"model", "messages", "tools"}:
                raise ValueError(f"extra_body cannot override {key}")
            body[key] = copy.deepcopy(value)
        return body

    def audit_metadata(self) -> dict[str, Any]:
        """Return safe provider labels suitable for a run manifest.

        In particular, this intentionally omits ``api_key`` and the full
        service URL path.  The endpoint host is enough to identify the
        transport without copying credentials into an artifact.
        """

        host = None
        if isinstance(self.config.base_url, str):
            host = urllib.parse.urlparse(self.config.base_url).netloc or None
        return {
            "transport": self.config.transport,
            "model": self.config.model,
            "endpoint_host": host,
            "service": self.config.service,
            "credential_configured": bool(self.config.api_key),
        }

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Send one completion and return its decoded OpenAI-style payload."""

        if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
            raise ValueError("messages must be a list of objects")
        if not isinstance(tools, list) or not all(isinstance(item, dict) for item in tools):
            raise ValueError("tools must be a list of objects")
        body = self._request_body(messages, tools, seed)
        if self.config.transport == "http":
            base_url = str(self.config.base_url).rstrip("/")
            url = f"{base_url}/chat/completions"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
            try:
                raw = self._http_post(url, body, headers, float(self.config.timeout))
            except ProviderAdapterError:
                raise
            except Exception as exc:
                raise ProviderAdapterError(
                    "provider HTTP request failed",
                    {"failure_class": "transport", "error_type": type(exc).__name__},
                ) from exc
            return _decode_payload(raw, source="HTTP")
        try:
            raw = self._trisol_exec(str(self.config.service), body, float(self.config.timeout))
        except ProviderAdapterError:
            raise
        except Exception as exc:
            raise ProviderAdapterError(
                "Trisol exec request failed",
                {"failure_class": "transport", "error_type": type(exc).__name__},
            ) from exc
        return _decode_payload(raw, source="Trisol")


def deepseek_provider(
    *,
    api_key: str,
    model: str = "deepseek-v4.1",
    base_url: str = "https://api.deepseek.com/v1",
    max_tokens: int = 5000,
) -> OpenAICompatibleProvider:
    """Construct the explicitly configured DeepSeek provider.

    The key is held only in memory for the request header and is never put in
    the provider metadata or exception text.
    """

    return OpenAICompatibleProvider(
        ProviderConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
    )


__all__ = [
    "OpenAICompatibleProvider",
    "ProviderAdapterError",
    "ProviderConfig",
    "deepseek_provider",
]
