"""Live provider test. Skipped unless --run-live is given and a key is configured."""

import pytest

from dgse_llm.llms import LLMRequest, MissingSecretError, build_llm, get_secret, parse_values
from dgse_llm.schemas import LLMConfig


@pytest.mark.live
def test_openai_round_trip():
    pytest.importorskip("openai")
    try:
        get_secret("OPENAI_API_KEY")
    except MissingSecretError:
        pytest.skip("no OPENAI_API_KEY configured")
    llm = build_llm(LLMConfig(name="o", provider="openai", model_id="gpt-4o-mini", cache=False))
    resp = llm.complete(LLMRequest('Reply with only this JSON: {"value": 1}'))
    assert parse_values(resp.text, ["value"]) == {"value": 1.0}
