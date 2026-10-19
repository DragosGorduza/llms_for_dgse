"""Deterministic offline provider for tests and dry runs.

Options (in LLMConfig.options):
  rules            : list of {pattern: <regex>, response: <str or dict>}; first match in the prompt wins
  default_response : used when no rule matches (default '{"value": 0.0}')
"""

from __future__ import annotations

import json
import re

from dgse_llm.schemas import LLMConfig

from ..base import LLMClient, LLMRequest, LLMResponse, register_llm


@register_llm("mock")
class MockLLM(LLMClient):
    provider = "mock"

    def __init__(self, config: LLMConfig) -> None:
        # Mock answers are free, so never write them to the shared disk cache.
        super().__init__(config.model_copy(update={"cache": False}))
        self.calls: list[str] = []

    def _complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request.prompt)
        opts = self.config.options
        for rule in opts.get("rules", []):
            if re.search(rule["pattern"], request.prompt, flags=re.IGNORECASE):
                return LLMResponse(text=self._as_text(rule["response"]))
        return LLMResponse(text=self._as_text(opts.get("default_response", {"value": 0.0})))

    @staticmethod
    def _as_text(response) -> str:
        return response if isinstance(response, str) else json.dumps(response)
