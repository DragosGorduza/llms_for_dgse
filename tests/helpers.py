"""Shared helpers for tests."""

from __future__ import annotations

import numpy as np

from dgse_llm.solvers import build_solver


def irf(model, T, deltas, solver=None):
    """Solve baseline and shocked paths; `deltas` maps exogenous variable -> additive deviation array."""
    solver = solver or build_solver("numpy_newton")
    base_exog = model.baseline_exog(T)
    exog = {k: v.copy() for k, v in base_exog.items()}
    for name, delta in deltas.items():
        exog[name] = exog[name] + np.asarray(delta, dtype=float)
    baseline = solver.solve(model, base_exog, T)
    shocked = solver.solve(model, exog, T)
    diff = {k: shocked.paths[k] - baseline.paths[k] for k in shocked.paths}
    return baseline, shocked, diff


def impulse(T, size, t=0):
    x = np.zeros(T)
    x[t] = size
    return x


NK_EVENTS = [{"t": 1, "text": "Energy prices have jumped 30% overnight."}, {"t": 6, "text": "Energy prices have fallen back."}]
NK_RULES = [
    {"pattern": "jumped", "response": {"inflation_expectation_change_pp": 2.0}},
    {"pattern": "fallen back", "response": {"inflation_expectation_change_pp": 0.0}},
]

# Answers by prompt type (the question sentence), for experiment specs; keyed like an experiment's `llms:` mapping.
MOCK_RULES = [
    {"pattern": "percentage change in total factor productivity", "response": {"change_pct": -2.0}},
    {"pattern": "percentage change in the household discount factor", "response": {"change_pct": 0.3}},
    {"pattern": "percentage change in household consumption", "response": {"change_pct": -15.0}},
    {"pattern": "percentage change in aggregate demand", "response": {"change_pct": -5.0}},
    {"pattern": "percentage change in firms' production costs", "response": {"change_pct": 10.0}},
    {"pattern": "percentage change in the central bank's policy interest rate", "response": {"change_pct": 20.0}},
    {"pattern": "percentage change in the inflation that households", "response": {"change_pct": 50.0}},
]


def mock_llms(**over):
    """An `llms:` mapping with one offline LLM called `mock` (temperature 0)."""
    setting = {"provider": "mock", "temperature": 0.0, "cache": False,
               "options": {"rules": MOCK_RULES, "default_response": {"change_pct": 0.0}}}
    return {"mock": {**setting, **over}}


def nk_experiment(id="exp", intensities=None, wedge_only=False, with_bad=False, repeats=1,
                  temperature=0.0, api_key_env=None):
    """Small experiment: manual cost-push (nk3) and TFP (rbc) shocks plus an LLM expectations wedge (nk3)."""
    from dgse_llm.schemas import (
        ExperimentConfig, LLMConfig, LLMSource, ModelConfig, Scenario, ShockSpec, Sweep, TimeProfile,
    )

    cost_push = ShockSpec(id="cp", variable="u", model="nk3", intensity=0.002,
                          profile=TimeProfile(kind="multi_period", duration=3))
    tfp = ShockSpec(id="tfp", variable="Z", model="rbc", intensity=0.01)
    wedge = ShockSpec(
        id="wedge", variable="wedge_pi", model="nk3", source="llm",
        profile=TimeProfile(kind="narrative_timeline"),
        llm=LLMSource(template="nk3/inflation_expectations.j2", keys=["inflation_expectation_change_pp"],
                      template_vars={"persona": "A tester.", "model_info": ""},
                      mapping="linear", mapping_params={"slope": 0.0025}, events=NK_EVENTS),
    )
    scenarios = [Scenario(id="wedge", shocks=[wedge])]
    if not wedge_only:
        scenarios.insert(0, Scenario(id="manual", shocks=[cost_push, tfp]))
    if with_bad:
        # `x` is an unknown of nk3, not an input, so building this shock must fail.
        scenarios.append(Scenario(id="bad", shocks=[ShockSpec(id="bad", variable="x", model="nk3")]))
    llm = LLMConfig(name="mock", provider="mock", temperature=temperature, cache=False,
                    api_key_env=api_key_env, options={"rules": NK_RULES})
    models = [ModelConfig(name="nk3")] if wedge_only else [ModelConfig(name="nk3"), ModelConfig(name="rbc")]
    return ExperimentConfig(
        id=id, T=30, models=models, llms=[llm], scenarios=scenarios,
        sweep=Sweep(intensities=intensities or [], repeats=repeats), tracked=["pi", "x", "i", "Y", "C"],
    )


def nk_arms_experiment(id="arms", classic_only=False, llm_only=False):
    """nk3 scenario with a classic cost-push shock and an LLM expectations wedge applied together (four arms)."""
    from dgse_llm.schemas import (
        ExperimentConfig, LLMConfig, LLMSource, ModelConfig, Scenario, ShockSpec, TimeProfile,
    )

    classic = ShockSpec(id="cp", source="classic", classic="cost_push_energy", model="nk3")
    wedge = ShockSpec(
        id="wedge", variable="wedge_pi", model="nk3", source="llm",
        profile=TimeProfile(kind="narrative_timeline"),
        llm=LLMSource(template="nk3/inflation_expectations.j2", keys=["inflation_expectation_change_pp"],
                      template_vars={"persona": "A tester.", "model_info": ""},
                      mapping="linear", mapping_params={"slope": 0.0025}, events=NK_EVENTS),
    )
    shocks = [classic] if classic_only else [wedge] if llm_only else [classic, wedge]
    llm = LLMConfig(name="mock", provider="mock", cache=False, options={"rules": NK_RULES})
    return ExperimentConfig(
        id=id, T=30, models=[ModelConfig(name="nk3")], llms=[llm],
        scenarios=[Scenario(id="energy", shocks=shocks)], tracked=["pi", "x", "i"],
    )
