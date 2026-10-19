"""Checks every registered model against the contract in models/base.py.

Expect: a new model dropped into src/dgse_llm/models is picked up here and must have a complete steady state,
residuals that vanish at the steady state, a zero response to a zero shock, and a .md documenting each variable.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.solvers import build_solver

SLOW = {"krusell_smith"}
PARAMS = [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in MODEL_REGISTRY.names()]


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_variable_declarations(name):
    cls = MODEL_REGISTRY.get(name)
    model = cls()
    names = [v.name for v in cls.variables]
    assert len(names) == len(set(names)), "duplicate variable names"
    assert len(model.unknowns) == len(cls.targets), "need one target per unknown"
    for v in cls.variables:
        assert v.role in ("exogenous", "unknown", "derived")
        if v.shockable:
            assert v.role == "exogenous", f"{v.name}: only exogenous inputs can be shockable"
    assert model.shockable, "a model needs at least one shockable input"


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_unknown_parameter_rejected(name):
    with pytest.raises(ValueError):
        MODEL_REGISTRY.get(name)({"not_a_parameter": 1.0})


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_docs_cover_every_variable_and_target(name):
    cls = MODEL_REGISTRY.get(name)
    md = Path(inspect.getfile(cls)).with_suffix(".md")
    assert md.exists(), f"{md.name} missing next to the model"
    text = md.read_text(encoding="utf-8")
    for n in [v.name for v in cls.variables] + list(cls.targets):
        assert f"### `{n}`" in text, f"{md.name} has no section '### `{n}`'"
    assert text.count("**If it rises:**") == len(cls.variables), "one 'If it rises' per variable"
    assert text.count("**If it falls:**") == len(cls.variables), "one 'If it falls' per variable"


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_every_variable_is_explained_in_code(name):
    for v in MODEL_REGISTRY.get(name).variables:
        assert v.units.strip(), f"{v.name}: units missing"
        assert len(v.description) >= 40, f"{v.name}: description should say what it is and its macro meaning"
        assert len(v.if_up) >= 20, f"{v.name}: if_up missing"
        assert len(v.if_down) >= 20, f"{v.name}: if_down missing"


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_classic_shocks_are_defined_and_valid(name):
    cls = MODEL_REGISTRY.get(name)
    model = cls()
    shocks = cls.classic_shocks
    assert len(shocks) >= 3, "define several classic shocks"
    assert len({s.name for s in shocks}) == len(shocks), "duplicate classic shock names"
    assert {s.kind for s in shocks} >= {"one_time", "multi_period"}, "need one-time and multi-period shocks"
    for s in shocks:
        assert s.variable in model.shockable, f"{s.name}: {s.variable} is not a shockable input"
        lo, hi = model.variable(s.variable).bounds
        assert s.size != 0 and abs(s.size) < 0.5 * (hi - lo), f"{s.name}: size looks wrong for the input's bounds"
        assert len(s.description) >= 40, f"{s.name}: describe the story in plain language"
        if s.kind == "multi_period":
            assert s.duration >= 2
        if s.kind == "geometric_decay":
            assert 0 < s.rho < 1
        assert model.classic_shock(s.name) is s
    with pytest.raises(KeyError, match="Available"):
        model.classic_shock("no_such_shock")


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_docs_describe_every_classic_shock(name):
    cls = MODEL_REGISTRY.get(name)
    text = Path(inspect.getfile(cls)).with_suffix(".md").read_text(encoding="utf-8")
    for s in cls.classic_shocks:
        assert f"### Classic shock `{s.name}`" in text, f"{name}.md has no section for classic shock `{s.name}`"
        assert f"| `{s.name}` |" in text, f"{name}.md table misses `{s.name}`"


@pytest.mark.parametrize("name", MODEL_REGISTRY.names())
def test_public_methods_have_docstrings(name):
    cls = MODEL_REGISTRY.get(name)
    for method in ("compute_steady_state", "residuals", "outputs"):
        doc = getattr(cls, method).__doc__ or ""
        assert len(doc) >= 60, f"{name}.{method} needs a docstring describing inputs and outputs"


@pytest.mark.parametrize("name", PARAMS)
def test_steady_state_complete_and_in_bounds(name, models):
    model = models[name]
    ss = model.steady_state
    for v in model.variables:
        assert v.name in ss and np.isfinite(ss[v.name])
        if v.shockable and v.bounds:
            assert v.bounds[0] < ss[v.name] < v.bounds[1]


@pytest.mark.parametrize("name", PARAMS)
def test_residuals_vanish_at_steady_state(name, models):
    model, T = models[name], 15
    U = {u: np.full(T, model.steady_state[u]) for u in model.unknowns}
    res = model.residuals(U, model.baseline_exog(T))
    assert set(res) == set(model.targets)
    for r in res.values():
        assert r.shape == (T,)
        assert np.max(np.abs(r)) < 1e-5


@pytest.mark.parametrize("name", PARAMS)
def test_zero_shock_stays_at_steady_state(name, models):
    model, T = models[name], 15
    sol = build_solver("numpy_newton").solve(model, model.baseline_exog(T), T)
    assert sol.diagnostics.converged
    assert set(sol.paths) == {v.name for v in model.variables}
    for k, path in sol.paths.items():
        assert path.shape == (T,)
        assert np.max(np.abs(path - model.steady_state[k])) < 1e-5, k
