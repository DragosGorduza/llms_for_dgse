"""Output record schemas: everything an experiment writes to disk."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import Mode


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class LLMCall(_Base):
    id: str
    shock_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    template: str
    prompt: str
    prompt_hash: str
    response_text: str = ""
    parsed: dict[str, float] | None = None
    provider: str
    model_id: str
    temperature: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float = 0.0
    cache_hit: bool = False
    timestamp: str
    error: str | None = None


class ShockPath(_Base):
    shock_id: str
    variable: str
    mode: Mode
    values: list[float]
    # In 'replace' mode only active periods overwrite the baseline input.
    active: list[bool]
    bounds_policy: Literal["error", "clip"] = "error"
    provenance: dict[str, Any] = Field(default_factory=dict)


class Diagnostics(_Base):
    solver: str
    converged: bool
    iterations: int
    residual_norm: float


class RunManifest(_Base):
    run_id: str
    experiment_id: str
    status: Literal["running", "ok", "failed"] = "running"
    error: str | None = None
    config_hash: str
    model: str
    llm: str | None = None
    scenario: str
    intensity: float | None = None
    mode: str | None = None
    repeat: int = 0
    # Experiment axes of the run (prompt, persona, scenario, model_info); empty for hand-written configs.
    meta: dict[str, Any] = Field(default_factory=dict)
    # The raw number the LLM answered (relative % change), first LLM shock; None without an LLM or after a failure.
    answer: float | None = None
    # One entry per modified variable: {"variable", "mode", "source", "shock_id"}; source is manual, classic or llm.
    modified: list[dict[str, Any]] = Field(default_factory=list)
    # Solved economies stored in trajectories.parquet: always baseline and shocked (all shocks applied); when the
    # scenario has both classic/manual and LLM shocks also classic (without the LLM) and llm (without the classic).
    arms: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Diagnostics] = Field(default_factory=dict)
    n_llm_calls: int = 0
    python: str = ""
    numpy: str = ""
    started: str = ""
    finished: str = ""
