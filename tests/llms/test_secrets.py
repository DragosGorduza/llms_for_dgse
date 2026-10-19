"""Secrets loading.

Expect: environment wins over the file, empty values count as missing, and a missing secret raises an error that
names the variable but never prints other secrets.
"""

import pytest

from dgse_llm.llms import MissingSecretError, get_secret


def test_reads_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DGSE_TEST_KEY", raising=False)
    f = tmp_path / "secrets.env"
    f.write_text("# comment\n\nDGSE_TEST_KEY='from_file'\nOTHER=zzz\n", encoding="utf-8")
    assert get_secret("DGSE_TEST_KEY", f) == "from_file"


def test_environment_overrides_file(tmp_path, monkeypatch):
    f = tmp_path / "secrets.env"
    f.write_text("DGSE_TEST_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("DGSE_TEST_KEY", "from_env")
    assert get_secret("DGSE_TEST_KEY", f) == "from_env"


def test_empty_value_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("DGSE_TEST_KEY", raising=False)
    f = tmp_path / "secrets.env"
    f.write_text("DGSE_TEST_KEY=\n", encoding="utf-8")
    with pytest.raises(MissingSecretError, match="DGSE_TEST_KEY"):
        get_secret("DGSE_TEST_KEY", f)


def test_missing_file_and_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("DGSE_TEST_KEY", raising=False)
    with pytest.raises(MissingSecretError) as err:
        get_secret("DGSE_TEST_KEY", tmp_path / "nope.env")
    assert "secrets/README.md" in str(err.value)


def test_openai_provider_fails_cleanly_without_key(tmp_path, monkeypatch):
    from dgse_llm.llms import LLMRequest, build_llm
    from dgse_llm.schemas import LLMConfig

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DGSE_SECRETS_DIR", str(tmp_path))
    llm = build_llm(LLMConfig(name="o", provider="openai", cache=False))
    with pytest.raises(MissingSecretError):
        llm.complete(LLMRequest("hi"))
