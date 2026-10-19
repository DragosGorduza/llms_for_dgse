"""LLM client interface, disk cache and registry."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from dgse_llm.paths import llm_cache_dir
from dgse_llm.registry import Registry
from dgse_llm.schemas import LLMConfig, stable_hash

LLM_REGISTRY = Registry("llm provider")
register_llm = LLM_REGISTRY.register


@dataclass
class LLMRequest:
    prompt: str
    # Differentiates repeated draws at temperature > 0 so they are not served from the cache.
    cache_salt: str = ""


class LLMResponse(BaseModel):
    """What a provider returns. Pydantic checks the types, so a provider that returns a non-string text, a negative
    latency or non-integer token counts fails immediately instead of corrupting the logs and the cache."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    text: StrictStr
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float = Field(0.0, ge=0)
    cache_hit: bool = False


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(json.dumps(value), encoding="utf-8")


class LLMClient(ABC):
    provider: ClassVar[str] = ""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        cache_dir = Path(config.cache_dir) if config.cache_dir else llm_cache_dir()
        # Only deterministic answers are reusable. With temperature > 0, or no temperature setting (reasoning models),
        # every call is a fresh draw from the response distribution and must not be replayed.
        deterministic = config.temperature == 0
        self._cache = DiskCache(cache_dir) if config.cache and deterministic else None

    def cache_key(self, request: LLMRequest) -> str:
        c = self.config
        return stable_hash(
            {
                "provider": c.provider,
                "model_id": c.model_id,
                "temperature": c.temperature,
                "seed": c.seed,
                "max_tokens": c.max_tokens,
                "prompt": request.prompt,
                "salt": request.cache_salt,
            },
            length=32,
        )

    def complete(self, request: LLMRequest) -> LLMResponse:
        key = self.cache_key(request) if self._cache else None
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return LLMResponse(**{**hit, "cache_hit": True})
        start = time.perf_counter()
        response = self._complete(request)
        response.latency_s = time.perf_counter() - start
        if self._cache is not None:
            self._cache.put(key, {**response.model_dump(), "cache_hit": False})
        return response

    @abstractmethod
    def _complete(self, request: LLMRequest) -> LLMResponse:
        """Call the provider. Must not log or return secrets."""


def build_llm(config: LLMConfig) -> LLMClient:
    return LLM_REGISTRY.get(config.provider)(config)
