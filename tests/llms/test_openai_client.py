"""OpenAI client construction without network access."""

import ssl
import sys
from types import ModuleType

from dgse_llm.llms.providers.openai import OpenAILLM
from dgse_llm.schemas import LLMConfig


def _install_openai_stub(monkeypatch, calls):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["openai"] = kwargs

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            calls["http_client"] = kwargs

    module = ModuleType("openai")
    module.OpenAI = FakeOpenAI
    module.DefaultHttpxClient = FakeHttpxClient
    monkeypatch.setitem(sys.modules, "openai", module)


def test_openai_client_uses_system_certificate_store(monkeypatch):
    calls = {}
    _install_openai_stub(monkeypatch, calls)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm = OpenAILLM(LLMConfig(name="o", provider="openai", cache=False))
    llm._get_client()

    assert isinstance(calls["http_client"]["verify"], ssl.SSLContext)
    assert calls["openai"]["api_key"] == "test-key"


def test_openai_client_accepts_custom_ca_bundle(monkeypatch, tmp_path):
    calls = {}
    _install_openai_stub(monkeypatch, calls)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    ca_bundle = str(tmp_path / "organization-ca.pem")
    config = LLMConfig(
        name="o", provider="openai", cache=False, options={"ca_bundle": ca_bundle}
    )

    OpenAILLM(config)._get_client()

    assert calls["http_client"]["verify"] == ca_bundle