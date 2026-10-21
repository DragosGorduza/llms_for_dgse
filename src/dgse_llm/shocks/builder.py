"""Turn ShockSpecs into paths and inject them into a model's exogenous inputs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from dgse_llm.llms import LLMClient, LLMRequest, ParseError, PromptRenderer, parse_values
from dgse_llm.models import Model
from dgse_llm.schemas import LLMCall, ShockPath, ShockSpec, TimeProfile, stable_hash

from .mapping import apply_mapping
from .profiles import make_shape

Recorder = Callable[[LLMCall], None]


class ShockError(ValueError):
    pass


def check_variable(spec: ShockSpec, model: Model) -> None:
    try:
        var = model.variable(spec.variable)
    except KeyError as exc:
        raise ShockError(f"Model '{model.name}' has no variable '{spec.variable}'") from exc
    if var.role != "exogenous" or not var.shockable:
        raise ShockError(
            f"Variable '{spec.variable}' of model '{model.name}' is not shockable. "
            f"Shockable: {model.shockable}"
        )


def resolve_classic(spec: ShockSpec, model: Model) -> ShockSpec:
    """Turn a `source: classic` spec into an equivalent manual spec using the model's own definition.

    The variable, timing and size come from the model's `ClassicShock`; `spec.intensity` multiplies the size and
    `spec.profile`, if set, overrides the timing. Returns a ShockSpec with source 'manual'.
    """
    try:
        classic = model.classic_shock(spec.classic)
    except KeyError as exc:
        raise ShockError(str(exc.args[0])) from exc
    if spec.variable not in (None, classic.variable):
        raise ShockError(
            f"Classic shock '{classic.name}' of model '{model.name}' modifies '{classic.variable}', "
            f"not '{spec.variable}'"
        )
    return spec.model_copy(
        update={
            "source": "manual",
            "classic": None,
            "variable": classic.variable,
            "profile": spec.profile or classic.profile(),
            "intensity": classic.size * spec.intensity,
        }
    )


def _ask_llm(spec, llm, renderer, event, t, recorder, salt) -> tuple[float, str]:
    src = spec.llm
    prompt = renderer.render(src.template, **{**src.template_vars, "narrative": src.narrative, "event": event, "t": t})
    base = dict(
        id=uuid.uuid4().hex[:10],
        shock_id=spec.id,
        context={"t": t},
        template=src.template,
        prompt=prompt,
        prompt_hash=stable_hash(prompt),
        provider=llm.config.provider,
        model_id=llm.config.model_id,
        temperature=llm.config.temperature,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    def record(**extra) -> None:
        if recorder is not None:
            recorder(LLMCall(**base, **extra))

    try:
        response = llm.complete(LLMRequest(prompt, cache_salt=salt))
    except Exception as exc:
        record(error=f"{type(exc).__name__}: {exc}")
        raise ShockError(f"LLM call failed for shock '{spec.id}': {exc}") from exc
    meta = dict(
        response_text=response.text,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        latency_s=response.latency_s,
        cache_hit=response.cache_hit,
    )
    try:
        parsed = parse_values(response.text, src.keys, src.answer_bounds)
    except ParseError as exc:
        record(error=f"ParseError: {exc}", **meta)
        raise ShockError(f"Could not parse LLM answer for shock '{spec.id}': {exc}") from exc
    record(parsed=parsed, **meta)
    return apply_mapping(src.mapping, parsed[src.key], src.mapping_params), base["id"]


def build_shock_path(
    spec: ShockSpec,
    model: Model,
    T: int,
    llm: LLMClient | None = None,
    renderer: PromptRenderer | None = None,
    recorder: Recorder | None = None,
    cache_salt: str = "",
) -> ShockPath:
    """Value path = intensity * (manual value, classic size or mapped LLM answer) * time profile."""
    classic_name = spec.classic if spec.source == "classic" else None
    if classic_name:
        spec = resolve_classic(spec, model)
    check_variable(spec, model)
    profile = spec.profile or TimeProfile()
    provenance: dict = {"source": "classic" if classic_name else spec.source}
    if classic_name:
        provenance["classic"] = classic_name
    if spec.source == "manual":
        shape = make_shape(T, profile)
        values = spec.intensity * shape
        active = shape != 0
    else:
        if llm is None:
            raise ShockError(f"Shock '{spec.id}' needs an LLM but none was configured")
        renderer = renderer or PromptRenderer()
        src = spec.llm
        # pct answer -> fraction of the input's baseline level, when the shock is relative to it.
        unit = 1.0
        if src.scale_by_baseline:
            level = float(model.steady_state[spec.variable])
            if level == 0.0:
                raise ShockError(
                    f"'{spec.variable}' of model '{model.name}' has baseline 0, so a relative change is undefined; "
                    "give a per_pct conversion instead"
                )
            unit = level
        memo: dict[str, tuple[float, str]] = {}
        call_ids: list[str] = []

        def ask(event: str, t: int) -> float:
            # One call per distinct event text, however many periods it covers.
            if event not in memo:
                memo[event] = _ask_llm(spec, llm, renderer, event, t, recorder, cache_salt)
                call_ids.append(memo[event][1])
            return memo[event][0] * unit

        if src.occurrences:
            values = np.zeros(T)
            active = np.zeros(T, dtype=bool)
            for occ in src.occurrences:
                v = ask(occ.text, min(occ.periods))
                for t in occ.periods:
                    if 0 <= t < T:
                        values[t] += spec.intensity * v
                        active[t] = True
        elif src.events:
            values = np.zeros(T)
            active = np.zeros(T, dtype=bool)
            t_last = src.events[-1].t
            for t in range(T):
                current = None
                for event in src.events:
                    if event.t <= t:
                        current = event
                if current is None:
                    continue
                v = ask(current.text, t)
                if current.t == t_last and t > t_last:
                    v *= src.tail_decay ** (t - t_last)
                values[t] = spec.intensity * v
                active[t] = True
        else:
            v = ask("", 0)
            shape = make_shape(T, profile)
            values = spec.intensity * v * shape
            active = shape != 0
        provenance.update(
            llm_call_ids=call_ids, mapping=src.mapping, mapping_params=src.mapping_params,
            scale_by_baseline=src.scale_by_baseline,
        )
    return ShockPath(
        shock_id=spec.id,
        variable=spec.variable,
        mode=spec.mode,
        values=[float(x) for x in values],
        active=[bool(x) for x in active],
        bounds_policy=spec.bounds_policy,
        provenance=provenance,
    )


def apply_shocks(model: Model, base_exog: dict[str, np.ndarray], paths: list[ShockPath]) -> dict[str, np.ndarray]:
    """additive: input = baseline + values; replace: input = values where active, baseline elsewhere."""
    exog = {k: v.copy() for k, v in base_exog.items()}
    for sp in paths:
        base = exog[sp.variable]
        values = np.asarray(sp.values, dtype=float)
        if len(values) != len(base):
            raise ShockError(f"Shock '{sp.shock_id}' has length {len(values)}, expected {len(base)}")
        if sp.mode == "additive":
            new = base + values
        else:
            new = np.where(np.asarray(sp.active, dtype=bool), values, base)
        bounds = model.variable(sp.variable).bounds
        if bounds is not None:
            lo, hi = bounds
            if np.any(new < lo) or np.any(new > hi):
                if sp.bounds_policy == "clip":
                    new = np.clip(new, lo, hi)
                else:
                    raise ShockError(
                        f"Shock '{sp.shock_id}' drives '{sp.variable}' outside [{lo}, {hi}] "
                        f"(range {new.min():.4g} to {new.max():.4g}); lower the intensity or set bounds_policy: clip"
                    )
        exog[sp.variable] = new
    return exog
