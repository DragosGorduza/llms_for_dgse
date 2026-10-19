"""Mock provider.

Expect: first matching rule wins (case-insensitive), dict responses are serialised to JSON, unmatched prompts get
the default response, prompts are recorded, and the shared disk cache is never used.
"""

import json

from dgse_llm.llms import LLMRequest, build_llm
from dgse_llm.schemas import LLMConfig


def mock(**options):
    return build_llm(LLMConfig(name="m", provider="mock", cache=True, options=options))


def test_first_matching_rule_wins_case_insensitive():
    llm = mock(rules=[{"pattern": "VACCINE", "response": {"v": 1}}, {"pattern": "vaccine", "response": {"v": 2}}])
    assert json.loads(llm.complete(LLMRequest("a vaccine arrived")).text) == {"v": 1}


def test_string_response_is_verbatim_and_default_applies():
    llm = mock(rules=[{"pattern": "x", "response": "plain text"}], default_response={"value": 7})
    assert llm.complete(LLMRequest("x")).text == "plain text"
    assert json.loads(llm.complete(LLMRequest("zzz")).text) == {"value": 7}


def test_builtin_default_response():
    assert json.loads(mock().complete(LLMRequest("anything")).text) == {"value": 0.0}


def test_prompts_recorded_and_never_cached(tmp_path):
    llm = build_llm(LLMConfig(name="m", provider="mock", cache=True, cache_dir=str(tmp_path)))
    llm.complete(LLMRequest("p"))
    r = llm.complete(LLMRequest("p"))
    assert llm.calls == ["p", "p"] and not r.cache_hit
    assert list(tmp_path.iterdir()) == []
