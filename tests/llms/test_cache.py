"""Disk cache behaviour of LLMClient.

Expect: at temperature 0 an identical request is served from the cache without calling the provider; a different
prompt, salt or model is a different entry; with temperature > 0 or no temperature (reasoning models) nothing is
cached, so every call is a fresh draw.
"""

import pytest

from dgse_llm.llms.base import LLMClient, LLMRequest, LLMResponse
from dgse_llm.schemas import LLMConfig


class Counting(LLMClient):
    provider = "counting"

    def __init__(self, config):
        super().__init__(config)
        self.n = 0

    def _complete(self, request):
        self.n += 1
        return LLMResponse(text=f"answer {self.n}", tokens_in=3, tokens_out=4)


def client(tmp_path, **kw):
    return Counting(LLMConfig(name="c", provider="counting", cache_dir=str(tmp_path), **kw))


def test_second_identical_call_is_a_cache_hit(tmp_path):
    c = client(tmp_path)
    first = c.complete(LLMRequest("hello"))
    second = c.complete(LLMRequest("hello"))
    assert c.n == 1
    assert not first.cache_hit and second.cache_hit
    assert second.text == first.text and second.tokens_in == 3


def test_cache_persists_across_clients(tmp_path):
    client(tmp_path).complete(LLMRequest("hello"))
    other = client(tmp_path)
    assert other.complete(LLMRequest("hello")).cache_hit and other.n == 0


def test_prompt_salt_and_model_change_the_key(tmp_path):
    c = client(tmp_path)
    c.complete(LLMRequest("hello"))
    c.complete(LLMRequest("hello!"))
    c.complete(LLMRequest("hello", cache_salt="rep1"))
    assert c.n == 3
    other_model = client(tmp_path, model_id="other")
    other_model.complete(LLMRequest("hello"))
    assert other_model.n == 1


def test_cache_can_be_disabled(tmp_path):
    c = client(tmp_path, cache=False)
    c.complete(LLMRequest("hello"))
    c.complete(LLMRequest("hello"))
    assert c.n == 2 and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("temperature", [0.7, 1.0, None])
def test_non_deterministic_settings_are_never_cached(tmp_path, temperature):
    c = client(tmp_path, temperature=temperature)
    a = c.complete(LLMRequest("hello"))
    b = c.complete(LLMRequest("hello"))
    assert c.n == 2 and not a.cache_hit and not b.cache_hit
    assert list(tmp_path.iterdir()) == []
