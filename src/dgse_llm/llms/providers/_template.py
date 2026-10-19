"""Template for a new LLM provider. Copy to <name>.py (no underscore), rename, and add configs/llms/<name>.yaml."""

from __future__ import annotations

from dgse_llm.schemas import LLMConfig

from ..base import LLMClient, LLMRequest, LLMResponse, register_llm
from ..secrets import get_secret


@register_llm("my_provider")
class MyProviderLLM(LLMClient):
    provider = "my_provider"

    def _complete(self, request: LLMRequest) -> LLMResponse:
        # Read the key lazily so offline tests never need it; use self.config for model_id, temperature, etc.
        key = get_secret(self.config.api_key_env or "MY_PROVIDER_API_KEY")
        raise NotImplementedError("Call the API with `key` and return LLMResponse(text=...)")
