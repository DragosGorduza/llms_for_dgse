"""Load an experiment file and compile it into the grid of scenarios the runner executes.

experiments/<model>/<name>.yaml (an `ExperimentSpec`) holds the whole setup: model, LLMs, prompts, personas, scenarios,
tracked variables. This module resolves the names it uses (personas, narratives, shock timelines from the
experiments folder) and expands the inner product

    prompts x personas x scenarios x include_model_info

into one single-shock `Scenario` each. The runner then adds LLMs and repeats.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from dgse_llm.paths import experiments_dir, prompts_dir
from dgse_llm.schemas import (
    ExperimentConfig,
    ExperimentSpec,
    LLMConfig,
    LLMSource,
    Occurrence,
    PromptDef,
    Scenario,
    ShockSpec,
    Sweep,
    load_yaml,
)

from .register import describe_model, load_register

ANSWER_KEY = "change_pct"


def load_prompt_table(model: str, root: Path | None = None) -> dict[str, PromptDef]:
    """prompts/<model>/prompts.yaml as {prompt kind: PromptDef}; every kind must have prompts/<model>/<kind>.j2."""
    folder = Path(root or prompts_dir()) / model
    path = folder / "prompts.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt table for model '{model}': expected {path}")
    table = {kind: PromptDef.model_validate(entry) for kind, entry in load_yaml(path).items()}
    for kind in table:
        if not (folder / f"{kind}.j2").exists():
            raise FileNotFoundError(f"prompts.yaml of '{model}' lists '{kind}' but {folder / (kind + '.j2')} is missing")
    return table


def _library(name: str, root: Path) -> dict[str, str]:
    path = root / f"{name}.yaml"
    return {str(k): str(v) for k, v in load_yaml(path).items()} if path.exists() else {}


def _period_keys(key: Any) -> list[int]:
    """A period key of a shock timeline: 4 is [4]; the string '4-7' is [4, 5, 6, 7]."""
    text = str(key).strip()
    if "-" in text:
        low, high = (int(part) for part in text.split("-", 1))
        if high < low:
            raise ValueError(f"period range '{text}' runs backwards")
        return list(range(low, high + 1))
    return [int(text)]


def load_shock_library(root: Path | None = None) -> dict[str, str]:
    """Every shock in experiments/shocks/<name>.yaml as {name: event text}.

    A shock is only an event description (`text:`, optional `description:`); it has no timing. Periods and the order of
    shocks are set in the scenarios of an experiment file.
    """
    folder = Path(root or experiments_dir()) / "shocks"
    library: dict[str, str] = {}
    for path in sorted(folder.glob("*.yaml")):
        data = load_yaml(path)
        extra = set(data) - {"text", "description"}
        if extra:
            raise ValueError(
                f"{path.name}: unexpected keys {sorted(map(str, extra))}. A shock has only `text`; set periods in the "
                "scenarios of the experiment file"
            )
        text = str(data.get("text", "")).strip()
        if not text:
            raise ValueError(f"{path.name}: a shock file needs a `text:` entry")
        library[path.stem] = text
    return library


def _scenario_events(name: str, timeline: Any, shocks: dict[str, str]) -> list[Occurrence]:
    """{5: shock_a, '8-10': [shock_b, shock_c], 12: shock_a} -> occurrences sorted by first period."""
    if not isinstance(timeline, dict) or not timeline:
        raise ValueError(
            f"Scenario '{name}': `shocks` must map periods to shock names, for example {{5: shock_a, 12: shock_b}}; "
            f"got {timeline!r}"
        )
    events: list[Occurrence] = []
    for key, value in timeline.items():
        names = [value] if isinstance(value, str) else list(value)
        missing = [n for n in names if n not in shocks]
        if missing:
            raise KeyError(f"Scenario '{name}' uses unknown shock {missing}. Available: {sorted(shocks)}")
        periods = _period_keys(key)
        events += [Occurrence(periods=periods, text=shocks[n], name=n) for n in names]
    return sorted(events, key=lambda o: o.periods[0])


def _pick(entry: Any, library: dict[str, str], what: str) -> dict[str, str]:
    """A mapping is used as is; 'all' is the whole library; a name or a list of names is looked up in it."""
    if isinstance(entry, dict):
        return {str(k): str(v) for k, v in entry.items()}
    if entry == "all":
        return dict(library)
    names = [entry] if isinstance(entry, str) else list(entry)
    missing = [n for n in names if n not in library]
    if missing:
        raise KeyError(f"Unknown {what} {missing}. Available: {sorted(library)}")
    return {n: library[n] for n in names}


def resolve_spec(raw: dict[str, Any], experiments_root: Path | None = None) -> ExperimentSpec:
    """Replace names by content: personas from personas.yaml, narratives and shocks from experiments/."""
    root = Path(experiments_root) if experiments_root else experiments_dir()
    raw = deepcopy(raw)
    raw["personas"] = _pick(raw.get("personas", "all"), _library("personas", root), "persona")
    narratives = _library("narratives", root)
    shocks = load_shock_library(root)
    scenarios = {}
    for name, entry in (raw.get("scenarios") or {}).items():
        entry = dict(entry or {})
        events = _scenario_events(name, entry.get("shocks"), shocks)
        entry["events"] = [o.model_dump() for o in events]
        entry["shocks"] = list(dict.fromkeys(o.name for o in events))
        context = str(entry.get("narrative") or "")
        entry["narrative"] = narratives.get(context, context)
        scenarios[name] = entry
    raw["scenarios"] = scenarios
    return ExperimentSpec.model_validate(raw)


def compile_spec(
    spec: ExperimentSpec, experiments_root: Path | None = None, prompts_root: Path | None = None
) -> ExperimentConfig:
    """Expand a spec into scenarios: one single-shock scenario per prompt x persona x scenario x info flag."""
    model = spec.model.name
    register = load_register(model, experiments_root)
    table = load_prompt_table(model, prompts_root)
    params = {**{k: p.default for k, p in register.parameters.items()}, **spec.model.params}
    llms = [LLMConfig(name=name, **setting.model_dump()) for name, setting in spec.llms.items()]
    scenarios: list[Scenario] = []
    for prompt in spec.prompts:
        if prompt not in table:
            raise KeyError(f"Model '{model}' has no prompt '{prompt}'. Available: {sorted(table)}")
        definition = table[prompt]
        variable = register.variables.get(definition.variable)
        if variable is None or not variable.shockable:
            raise ValueError(f"Prompt '{prompt}' targets '{definition.variable}', which is not a shockable input of '{model}'")
        info = describe_model(register, params, definition.variable)
        relative = definition.per_pct is None
        for persona_id, persona in spec.personas.items():
            for scenario_name, scenario in spec.scenarios.items():
                for with_info in spec.include_model_info:
                    scenario_id = ".".join([prompt, persona_id, scenario_name, "info" if with_info else "noinfo"])
                    shock = ShockSpec(
                        id=scenario_id, model=model, variable=definition.variable, source="llm",
                        llm=LLMSource(
                            template=f"{model}/{prompt}.j2",
                            keys=[ANSWER_KEY],
                            mapping="pct" if relative else "linear",
                            mapping_params={} if relative else {"slope": definition.per_pct},
                            scale_by_baseline=relative,
                            narrative=scenario.narrative,
                            occurrences=scenario.events,
                            template_vars={"persona": persona, "model_info": info if with_info else ""},
                            answer_bounds={ANSWER_KEY: definition.answer_bounds},
                        ),
                    )
                    scenarios.append(Scenario(
                        id=scenario_id,
                        meta={"prompt": prompt, "persona": persona_id, "scenario": scenario_name,
                              "model_info": with_info},
                        shocks=[shock],
                    ))
    return ExperimentConfig(
        id=spec.id, description=spec.description, T=spec.T, models=[spec.model], llms=llms,
        scenarios=scenarios, sweep=Sweep(repeats=spec.repeats), tracked=spec.tracked, out_dir=spec.out_dir,
    )


def resolve_experiment(
    raw: dict[str, Any], experiments_root: Path | None = None, prompts_root: Path | None = None,
) -> ExperimentConfig:
    return compile_spec(resolve_spec(raw, experiments_root), experiments_root, prompts_root)


def load_experiment(path: str | Path, experiments_root: Path | None = None) -> ExperimentConfig:
    """Read an experiment YAML (an `ExperimentSpec`) and compile it into an `ExperimentConfig`."""
    return resolve_experiment(load_yaml(path), experiments_root)
