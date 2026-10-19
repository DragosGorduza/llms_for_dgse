"""Input configuration schemas: which models, LLMs, shocks and sweeps an experiment uses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["additive", "replace"]
ProfileKind = Literal["one_time", "multi_period", "geometric_decay", "from_series", "narrative_timeline"]


class _Base(BaseModel):
    # protected_namespaces=() allows field names such as `model_id`
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(_Base):
    name: str
    params: dict[str, float] = Field(default_factory=dict)
    solver: str = "numpy_newton"
    solver_options: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(_Base):
    name: str
    provider: str
    model_id: str = "mock"
    temperature: float | None = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    # Name of the environment variable / secrets.env entry holding the key; never the key itself.
    api_key_env: str | None = None
    # Caching only applies at temperature 0; any other value (including None) always calls the provider.
    cache: bool = True
    cache_dir: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class TimeProfile(_Base):
    kind: ProfileKind = "one_time"
    start: int = Field(0, ge=0)
    duration: int = Field(1, ge=1)
    rho: float = 0.9
    series: list[float] | None = None


class NarrativeEvent(_Base):
    t: int = Field(ge=0)
    text: str


class LLMSource(_Base):
    template: str
    keys: list[str] = Field(min_length=1)
    key: str | None = None
    mapping: str = "identity"
    mapping_params: dict[str, float] = Field(default_factory=dict)
    template_vars: dict[str, Any] = Field(default_factory=dict)
    # Plausibility range (low, high) of the RAW answer per key, in the units the prompt asks for. An answer outside it
    # is rejected as invalid instead of being fed to the model.
    answer_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    events: list[NarrativeEvent] = Field(default_factory=list)
    # After the last event the value is multiplied by tail_decay ** (t - t_last).
    tail_decay: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> "LLMSource":
        if self.key is None:
            self.key = self.keys[0]
        if self.key not in self.keys:
            raise ValueError(f"key '{self.key}' must be one of keys {self.keys}")
        for name, (lo, hi) in self.answer_bounds.items():
            if name not in self.keys:
                raise ValueError(f"answer_bounds names '{name}', which is not one of keys {self.keys}")
            if lo >= hi:
                raise ValueError(f"answer_bounds for '{name}' must have low < high, got ({lo}, {hi})")
        self.events = sorted(self.events, key=lambda e: e.t)
        return self


class ShockSpec(_Base):
    """One shock on one model input.

    source 'manual'  : you give `variable`, `profile` and `intensity` (the value in input units).
    source 'classic' : a named shock defined by the model itself (see `classic_shocks` in the model's .py and .md).
                       `variable`, size and timing come from the model; `intensity` is a MULTIPLIER of the model's
                       size (2 = twice as big, -1 = the opposite shock); `profile`, if given, overrides the timing.
    source 'llm'     : the value comes from an LLM answer (see `llm`); `intensity` multiplies the mapped answer.
    """

    id: str
    variable: str | None = None
    # If set, the shock only applies to this model; other models skip it.
    model: str | None = None
    mode: Mode = "additive"
    profile: TimeProfile | None = None
    intensity: float = 1.0
    source: Literal["manual", "classic", "llm"] = "manual"
    classic: str | None = None
    llm: LLMSource | None = None
    bounds_policy: Literal["error", "clip"] = "error"

    @model_validator(mode="after")
    def _check(self) -> "ShockSpec":
        if self.source == "classic":
            if not self.classic:
                raise ValueError("source 'classic' requires `classic`: the name of a shock defined by the model")
            if self.llm is not None:
                raise ValueError("source 'classic' must not have an 'llm' block")
        else:
            if self.variable is None:
                raise ValueError(f"source '{self.source}' requires `variable`")
            if self.classic is not None:
                raise ValueError("`classic` is only allowed with source 'classic'")
        if self.source == "llm" and self.llm is None:
            raise ValueError("source 'llm' requires an 'llm' block")
        if self.source == "manual" and self.llm is not None:
            raise ValueError("source 'manual' must not have an 'llm' block")
        profile = self.profile or TimeProfile()
        narrative = profile.kind == "narrative_timeline"
        if narrative and not (self.llm and self.llm.events):
            raise ValueError("profile 'narrative_timeline' requires llm.events")
        if self.llm and self.llm.events and not narrative:
            raise ValueError("llm.events requires profile kind 'narrative_timeline'")
        if profile.kind == "from_series" and not profile.series:
            raise ValueError("profile 'from_series' requires a series")
        return self


class Scenario(_Base):
    id: str
    description: str = ""
    shocks: list[ShockSpec] = Field(min_length=1)


class Sweep(_Base):
    # Empty lists mean "use the value written in each ShockSpec".
    intensities: list[float] = Field(default_factory=list)
    modes: list[Mode] = Field(default_factory=list)
    repeats: int = Field(1, ge=1)


class ExperimentConfig(_Base):
    id: str
    description: str = ""
    T: int = Field(100, gt=1)
    models: list[ModelConfig] = Field(min_length=1)
    llms: list[LLMConfig] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(min_length=1)
    sweep: Sweep = Field(default_factory=Sweep)
    tracked: list[str] | None = None
    out_dir: str | None = None
