"""OpenAI provider (Responses API). Needs `pip install openai` and OPENAI_API_KEY (see secrets/README.md)."""

from __future__ import annotations

from dgse_llm.schemas import LLMConfig

from ..base import LLMClient, LLMRequest, LLMResponse, register_llm
from ..secrets import get_secret


@register_llm("openai")
class OpenAILLM(LLMClient):
    provider = "openai"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            key = get_secret(self.config.api_key_env or "OPENAI_API_KEY")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("The 'openai' package is required: pip install openai") from exc
            self._client = OpenAI(api_key=key)
        return self._client

    def _complete(self, request: LLMRequest) -> LLMResponse:
        c = self.config
        kwargs: dict = {"model": c.model_id, "input": request.prompt}
        if c.temperature is not None:
            kwargs["temperature"] = c.temperature
        if c.max_tokens is not None:
            kwargs["max_output_tokens"] = c.max_tokens
        # The Responses API has no seed parameter, so c.seed is not sent.
        response = self._get_client().responses.create(**kwargs)
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=response.output_text,
            tokens_in=getattr(usage, "input_tokens", None),
            tokens_out=getattr(usage, "output_tokens", None),
        )
