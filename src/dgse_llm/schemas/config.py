"""Input configuration schemas: which models, LLMs, shocks and sweeps an experiment uses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["additive", "replace"]
ProfileKind = Literal[
    "one_time", "multi_period", "geometric_decay", "from_series", "narrative_timeline", "at_periods"
]


class _Base(BaseModel):
    # protected_namespaces=() allows field names such as `model_id`
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelConfig(_Base):
    name: str
    params: dict[str, float] = Field(default_factory=dict)
    solver: str = "numpy_newton"
    solver_options: dict[str, Any] = Field(default_factory=dict)


class LLMSetting(_Base):
    """One LLM as written in an experiment's `llms:` mapping (the key is its name)."""

    provider: str
    model_id: str = "mock"
    # null: the model takes no temperature (for example reasoning models); every call is then a fresh draw.
    temperature: float | None = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    # Name of the environment variable / secrets.env entry holding the key; never the key itself. null: no key needed.
    api_key_env: str | None = None
    # Caching only applies at temperature 0; any other value (including None) always calls the provider.
    cache: bool = True
    cache_dir: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(LLMSetting):
    name: str


class TimeProfile(_Base):
    kind: ProfileKind = "one_time"
    start: int = Field(0, ge=0)
    duration: int = Field(1, ge=1)
    rho: float = 0.9
    series: list[float] | None = None
    # kind 'at_periods': the shock is active exactly in these periods (one entry = a single-period shock).
    periods: list[int] = Field(default_factory=list)


class NarrativeEvent(_Base):
    t: int = Field(ge=0)
    text: str


class Occurrence(_Base):
    """An event that happens in one or more periods: one LLM call for `text`, its value applied in each period."""

    periods: list[int] = Field(min_length=1)
    text: str
    # Name of the shock (experiments/shocks/<name>.yaml) the text came from; empty for hand-written events.
    name: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Occurrence":
        if any(p < 0 for p in self.periods):
            raise ValueError(f"periods must be >= 0, got {self.periods}")
        return self


class LLMSource(_Base):
    template: str
    keys: list[str] = Field(min_length=1)
    key: str | None = None
    mapping: str = "identity"
    mapping_params: dict[str, float] = Field(default_factory=dict)
    template_vars: dict[str, Any] = Field(default_factory=dict)
    # Narrative text used when there are no `events` (one LLM call, applied in every active period).
    narrative: str = ""
    # Broader context shown with every event (template slot `narrative`); events fill the slot `event`.
    # With `occurrences`, each event is one call and its answer is applied in the listed periods only (no holding).
    occurrences: list[Occurrence] = Field(default_factory=list)
    # True: the mapped answer is a fraction of the input's baseline level (answer / 100 * baseline; use mapping 'pct').
    scale_by_baseline: bool = False
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
        if self.llm and self.llm.occurrences and self.llm.events:
            raise ValueError("llm.occurrences and llm.events cannot be combined")
        if profile.kind == "from_series" and not profile.series:
            raise ValueError("profile 'from_series' requires a series")
        if profile.kind == "at_periods" and not profile.periods:
            raise ValueError("profile 'at_periods' requires periods")
        if self.llm and self.llm.scale_by_baseline and self.mode != "additive":
            raise ValueError("scale_by_baseline needs mode 'additive'")
        return self


class Scenario(_Base):
    id: str
    description: str = ""
    # Experiment axes of this scenario (prompt, persona, scenario, model_info), copied to the run manifest.
    meta: dict[str, Any] = Field(default_factory=dict)
    shocks: list[ShockSpec] = Field(min_length=1)


class Sweep(_Base):
    # Empty lists mean "use the value written in each ShockSpec".
    intensities: list[float] = Field(default_factory=list)
    modes: list[Mode] = Field(default_factory=list)
    repeats: int = Field(1, ge=1)


class ExperimentConfig(_Base):
    """The expanded plan the runner executes; users write an `ExperimentSpec`, which compiles into this."""

    id: str
    description: str = ""
    T: int = Field(100, gt=1)
    models: list[ModelConfig] = Field(min_length=1)
    llms: list[LLMConfig] = Field(default_factory=list)
    scenarios: list[Scenario] = Field(min_length=1)
    sweep: Sweep = Field(default_factory=Sweep)
    tracked: list[str] | None = None
    out_dir: str | None = None


class ScenarioDef(_Base):
    """One named scenario of an experiment: an optional broader narrative plus one or more named shocks."""

    description: str = ""
    # Broader context: a name from experiments/narratives.yaml or literal text. Empty: no context line in the prompt.
    narrative: str = ""
    # In the YAML: {period: shock name or list of names}, e.g. {5: shock_a, 12: shock_b}; "8-10" keys are ranges.
    # The loader turns it into `events` and keeps here the list of shock names used.
    shocks: list[str] = Field(min_length=1)
    events: list[Occurrence] = Field(default_factory=list)


class ExperimentSpec(_Base):
    """What a user writes in experiments/<model>/<name>.yaml (names already resolved by the loader).

    Cells = llms x prompts x personas x scenarios x include_model_info (x repeats when temperature != 0).
    """

    id: str
    description: str = ""
    T: int = Field(120, gt=1)
    # The macro model, with optional parameter overrides and solver settings; a bare name works too.
    model: ModelConfig
    # name -> settings. Any number of entries; the same provider can appear several times (for example at
    # different temperatures), and every entry is run on every cell.
    llms: dict[str, LLMSetting] = Field(min_length=1)
    # Prompt kinds: file stems in prompts/<model>/ (each shocks one input, see prompts/<model>/prompts.yaml).
    prompts: list[str] = Field(min_length=1)
    # id -> text. Who answers.
    personas: dict[str, str] = Field(min_length=1)
    # id -> narrative (optional) and shocks; each shock is a timeline {period: event text}.
    scenarios: dict[str, ScenarioDef] = Field(min_length=1)
    # Run every cell with and/or without the model's name, description and parameters in the prompt.
    include_model_info: list[bool] = Field(default_factory=lambda: [False, True], min_length=1)
    # Draws per cell for LLMs with temperature != 0; temperature 0 is deterministic, so one draw.
    repeats: int = Field(5, ge=1)
    # Variables saved and reported; null means all.
    tracked: list[str] | None = None
    out_dir: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _bare_model_name(cls, data):
        if isinstance(data, dict) and isinstance(data.get("model"), str):
            data = {**data, "model": {"name": data["model"]}}
        return data

    @model_validator(mode="after")
    def _check(self) -> "ExperimentSpec":
        for name, scenario in self.scenarios.items():
            if not scenario.events:
                raise ValueError(f"scenario '{name}' has no events")
            late = [p for occ in scenario.events for p in occ.periods if p >= self.T]
            if late:
                raise ValueError(f"scenario '{name}' has periods {sorted(set(late))} outside 0..{self.T - 1}")
        if len(set(self.include_model_info)) != len(self.include_model_info):
            raise ValueError("include_model_info lists a value twice")
        return self


class PromptDef(_Base):
    """One entry of prompts/<model>/prompts.yaml: which input a prompt kind shocks and how its answer is converted.

    Every prompt asks for the same thing, a relative percentage change (JSON key `change_pct`).
    """

    variable: str
    # Plausible range of the answer in percent; anything outside is rejected as invalid.
    answer_bounds: tuple[float, float]
    # Input units per +1% answered. None: the answer is a relative change of the input's own baseline level.
    per_pct: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "PromptDef":
        if self.answer_bounds[0] >= self.answer_bounds[1]:
            raise ValueError(f"answer_bounds must have low < high, got {self.answer_bounds}")
        if self.per_pct == 0:
            raise ValueError("per_pct must not be 0")
        return self


class ParamInfo(_Base):
    default: float
    description: str = Field(min_length=10)


class VarInfo(_Base):
    role: Literal["exogenous", "unknown", "derived"]
    shockable: bool
    units: str = Field(min_length=3)
    description: str = Field(min_length=20)


class ModelRegister(_Base):
    """experiments/<model>/model.json: the model's description, every parameter and every variable."""

    name: str
    title: str
    description: str = Field(min_length=40)
    parameters: dict[str, ParamInfo]
    variables: dict[str, VarInfo]
