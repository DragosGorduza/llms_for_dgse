"""Type checking of what providers return.

Expect: LLMResponse is a pydantic model that rejects a non-string text, non-integer token counts, a negative latency and
unknown fields, so a misbehaving provider fails loudly at once instead of corrupting logs or the cache.
"""

import pytest
from pydantic import ValidationError

from dgse_llm.llms.base import LLMClient, LLMRequest, LLMResponse
from dgse_llm.schemas import LLMConfig


def test_valid_response_defaults():
    r = LLMResponse(text="hello")
    assert r.tokens_in is None and r.latency_s == 0.0 and r.cache_hit is False


@pytest.mark.parametrize("bad", [None, 3, ["a"], b"bytes"])
def test_text_must_be_a_string(bad):
    with pytest.raises(ValidationError):
        LLMResponse(text=bad)


def test_token_counts_must_be_integers():
    with pytest.raises(ValidationError):
        LLMResponse(text="x", tokens_in="many")
    with pytest.raises(ValidationError):
        LLMResponse(text="x", tokens_out=2.5)
    assert LLMResponse(text="x", tokens_in=3, tokens_out=4).tokens_out == 4


def test_latency_cannot_be_negative_and_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        LLMResponse(text="x", latency_s=-1.0)
    with pytest.raises(ValidationError):
        LLMResponse(text="x", surprise=1)


def test_assignment_is_validated():
    r = LLMResponse(text="x")
    with pytest.raises(ValidationError):
        r.latency_s = "slow"


class BadProvider(LLMClient):
    provider = "bad"

    def _complete(self, request):
        return LLMResponse(text=None)


def test_client_surfaces_a_provider_returning_the_wrong_type(tmp_path):
    client = BadProvider(LLMConfig(name="b", provider="bad", cache_dir=str(tmp_path)))
    with pytest.raises(ValidationError):
        client.complete(LLMRequest("hi"))
    assert list(tmp_path.iterdir()) == [], "nothing invalid may be cached"
