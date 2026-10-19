"""Building shock paths and injecting them.

Expect:
- manual shocks: value = intensity x profile; additive adds to the baseline, replace overwrites only active periods
- inputs must be shockable and inside the variable's bounds (error, or clipped with bounds_policy=clip)
- LLM shocks: mapped answer x intensity x profile; narrative timelines call the LLM once per distinct narrative,
  hold each answer until the next event, and decay after the last event
- every LLM call is recorded, failures are recorded and raised, provenance links shock paths to calls
"""

import numpy as np
import pytest

from dgse_llm.llms import build_llm
from dgse_llm.schemas import LLMConfig, LLMSource, ShockSpec, TimeProfile
from dgse_llm.shocks import ShockError, apply_shocks, build_shock_path

T = 12


def manual(variable="u", **kw):
    return ShockSpec(id="s", variable=variable, **kw)


def mock_llm(**options):
    return build_llm(LLMConfig(name="m", provider="mock", options=options))


def llm_spec(variable="wedge_pi", events=None, tail_decay=1.0, intensity=1.0, mode="additive", profile=None):
    src = LLMSource(
        template="inflation_expectations.j2",
        keys=["inflation_expectation_change_pp"],
        mapping="linear",
        mapping_params={"slope": 0.01},
        events=events or [],
        tail_decay=tail_decay,
    )
    if profile is None:
        profile = TimeProfile(kind="narrative_timeline" if events else "one_time")
    return ShockSpec(id="w", variable=variable, source="llm", llm=src, profile=profile,
                     intensity=intensity, mode=mode)


# ---- manual shocks -------------------------------------------------------------------------------------------

def test_manual_additive_one_time(models):
    m = models["nk3"]
    path = build_shock_path(manual(intensity=0.01, profile=TimeProfile(kind="one_time", start=2)), m, T)
    assert path.values[2] == pytest.approx(0.01) and sum(path.values) == pytest.approx(0.01)
    assert path.active[2] and sum(path.active) == 1
    exog = apply_shocks(m, m.baseline_exog(T), [path])
    assert exog["u"][2] == pytest.approx(0.01) and exog["u"][3] == 0
    assert exog["r_n"].sum() == 0, "other inputs must be untouched"


def test_additive_stacks_on_a_non_steady_state_baseline(models):
    m = models["nk3"]
    base = m.baseline_exog(T)
    base["u"] = base["u"] + 0.1
    path = build_shock_path(manual(intensity=0.01), m, T)
    assert apply_shocks(m, base, [path])["u"][0] == pytest.approx(0.11)


def test_replace_overwrites_only_active_periods(models):
    m = models["nk3"]
    base = m.baseline_exog(T)
    base["u"] = base["u"] + 0.1
    spec = manual(intensity=0.02, mode="replace", profile=TimeProfile(kind="multi_period", start=1, duration=3))
    exog = apply_shocks(m, base, [build_shock_path(spec, m, T)])
    assert np.allclose(exog["u"][1:4], 0.02)
    assert exog["u"][0] == pytest.approx(0.1) and exog["u"][4] == pytest.approx(0.1)


def test_does_not_mutate_baseline(models):
    m = models["nk3"]
    base = m.baseline_exog(T)
    apply_shocks(m, base, [build_shock_path(manual(intensity=0.01), m, T)])
    assert base["u"].sum() == 0


def test_multiple_shocks_on_different_variables(models):
    m = models["nk3"]
    paths = [
        build_shock_path(manual("u", intensity=0.01), m, T),
        build_shock_path(manual("eps_i", intensity=-0.02), m, T),
    ]
    exog = apply_shocks(m, m.baseline_exog(T), paths)
    assert exog["u"][0] == pytest.approx(0.01) and exog["eps_i"][0] == pytest.approx(-0.02)


# ---- validation ----------------------------------------------------------------------------------------------

def test_non_shockable_and_unknown_variables_rejected(models):
    m = models["rbc"]
    with pytest.raises(ShockError, match="not shockable"):
        build_shock_path(manual("K"), m, T)
    with pytest.raises(ShockError, match="no variable"):
        build_shock_path(manual("nonsense"), m, T)


def test_bounds_error_and_clip(models):
    m = models["rbc"]
    spec = manual("Z", intensity=-2.0)
    with pytest.raises(ShockError, match="outside"):
        apply_shocks(m, m.baseline_exog(T), [build_shock_path(spec, m, T)])
    clipped = apply_shocks(m, m.baseline_exog(T), [build_shock_path(spec.model_copy(update={"bounds_policy": "clip"}), m, T)])
    assert clipped["Z"].min() == pytest.approx(m.variable("Z").bounds[0])


# ---- classic (model-defined) shocks --------------------------------------------------------------------------

def classic(name, **kw):
    return ShockSpec(id="c", source="classic", classic=name, **kw)


def test_classic_one_time_uses_model_definition(models):
    m = models["rbc"]
    path = build_shock_path(classic("tfp_slump_one_time"), m, T)
    assert path.variable == "Z" and path.provenance == {"source": "classic", "classic": "tfp_slump_one_time"}
    assert path.values[0] == pytest.approx(-0.02) and sum(path.active) == 1


def test_classic_multi_period_and_decay(models):
    m = models["nk3"]
    multi = build_shock_path(classic("cost_push_energy"), m, T)
    assert multi.variable == "u" and multi.values[:5] == pytest.approx([0.002] * 4 + [0.0])
    decay = build_shock_path(classic("tfp_boom_persistent"), models["rbc"], T)
    assert decay.values[:3] == pytest.approx([0.01, 0.009, 0.0081])


def test_classic_intensity_is_a_multiplier(models):
    m = models["nk3"]
    twice = build_shock_path(classic("monetary_tightening", intensity=2.0), m, T)
    flipped = build_shock_path(classic("monetary_tightening", intensity=-1.0), m, T)
    assert twice.values[0] == pytest.approx(0.005) and flipped.values[0] == pytest.approx(-0.0025)


def test_classic_profile_can_be_overridden(models):
    m = models["nk3"]
    path = build_shock_path(
        classic("cost_push_energy", profile=TimeProfile(kind="multi_period", start=2, duration=2)), m, T
    )
    assert path.values[:5] == pytest.approx([0, 0, 0.002, 0.002, 0])


def test_classic_unknown_name_or_wrong_variable_rejected(models):
    m = models["nk3"]
    with pytest.raises(ShockError, match="Available"):
        build_shock_path(classic("nope"), m, T)
    with pytest.raises(ShockError, match="modifies 'u'"):
        build_shock_path(classic("cost_push_energy", variable="eps_i"), m, T)


def test_classic_shock_applies_like_a_manual_one(models):
    m = models["nk3"]
    exog = apply_shocks(m, m.baseline_exog(T), [build_shock_path(classic("demand_slump"), m, T)])
    assert np.allclose(exog["r_n"][:4], -0.005) and exog["r_n"][4] == 0


# ---- LLM shocks ----------------------------------------------------------------------------------------------

def test_llm_single_call_maps_and_scales(models):
    m, llm, calls = models["nk3"], mock_llm(default_response={"inflation_expectation_change_pp": 2.0}), []
    path = build_shock_path(llm_spec(intensity=3.0), m, T, llm=llm, recorder=calls.append)
    assert path.values[0] == pytest.approx(0.01 * 2.0 * 3.0) and sum(path.values[1:]) == 0
    assert len(calls) == 1 and calls[0].parsed == {"inflation_expectation_change_pp": 2.0}
    assert calls[0].shock_id == "w" and calls[0].provider == "mock" and not calls[0].error
    assert path.provenance["llm_call_ids"] == [calls[0].id]


def test_llm_shock_without_client_rejected(models):
    with pytest.raises(ShockError, match="needs an LLM"):
        build_shock_path(llm_spec(), models["nk3"], T)


EVENTS = [{"t": 2, "text": "Energy prices JUMPED."}, {"t": 6, "text": "Energy prices FELL."}]
RULES = [
    {"pattern": "JUMPED", "response": {"inflation_expectation_change_pp": 4.0}},
    {"pattern": "FELL", "response": {"inflation_expectation_change_pp": 1.0}},
]


def test_narrative_timeline_holds_each_answer_until_next_event(models):
    m, llm, calls = models["nk3"], mock_llm(rules=RULES), []
    path = build_shock_path(llm_spec(events=EVENTS), m, T, llm=llm, recorder=calls.append)
    v = np.array(path.values)
    assert np.allclose(v[:2], 0) and np.allclose(v[2:6], 0.04) and np.allclose(v[6:], 0.01)
    assert path.active == [False, False] + [True] * 10
    assert len(calls) == 2 and len(llm.calls) == 2, "one call per distinct narrative"
    assert [c.context["t"] for c in calls] == [2, 6]


def test_narrative_tail_decay_applies_after_last_event_only(models):
    m, llm = models["nk3"], mock_llm(rules=RULES)
    path = build_shock_path(llm_spec(events=EVENTS, tail_decay=0.5), m, T, llm=llm)
    v = np.array(path.values)
    assert np.allclose(v[2:6], 0.04), "earlier events are not decayed"
    assert np.allclose(v[6:9], [0.01, 0.005, 0.0025])


def test_repeated_narrative_is_asked_once(models):
    m, llm = models["nk3"], mock_llm(default_response={"inflation_expectation_change_pp": 1.0})
    events = [{"t": 0, "text": "same"}, {"t": 4, "text": "same"}]
    build_shock_path(llm_spec(events=events), m, T, llm=llm)
    assert len(llm.calls) == 1


def test_narrative_replace_mode_only_overwrites_active_periods(models):
    m, llm = models["nk3"], mock_llm(rules=RULES)
    path = build_shock_path(llm_spec(events=EVENTS, mode="replace"), m, T, llm=llm)
    base = m.baseline_exog(T)
    base["wedge_pi"] = base["wedge_pi"] + 0.5
    exog = apply_shocks(m, base, [path])
    assert np.allclose(exog["wedge_pi"][:2], 0.5) and np.allclose(exog["wedge_pi"][2:6], 0.04)


def test_prompt_contains_narrative_and_is_recorded(models):
    m, llm, calls = models["nk3"], mock_llm(rules=RULES), []
    build_shock_path(llm_spec(events=EVENTS), m, T, llm=llm, recorder=calls.append)
    assert "Energy prices JUMPED." in calls[0].prompt and calls[0].prompt_hash


def test_unparseable_answer_is_recorded_then_raised(models):
    m, llm, calls = models["nk3"], mock_llm(default_response="I cannot answer."), []
    with pytest.raises(ShockError, match="parse"):
        build_shock_path(llm_spec(), m, T, llm=llm, recorder=calls.append)
    assert len(calls) == 1 and calls[0].error and calls[0].response_text == "I cannot answer."


def test_missing_key_in_answer_is_an_error(models):
    m, llm, calls = models["nk3"], mock_llm(default_response={"other": 1}), []
    with pytest.raises(ShockError):
        build_shock_path(llm_spec(), m, T, llm=llm, recorder=calls.append)
    assert "Missing keys" in calls[0].error


def test_answer_outside_declared_bounds_is_rejected_and_recorded(models):
    m, calls = models["nk3"], []
    llm = mock_llm(default_response={"inflation_expectation_change_pp": 900.0})
    spec = llm_spec()
    spec.llm.answer_bounds = {"inflation_expectation_change_pp": (-50.0, 50.0)}
    with pytest.raises(ShockError, match="parse"):
        build_shock_path(spec, m, T, llm=llm, recorder=calls.append)
    assert len(calls) == 1 and "less than or equal to" in calls[0].error and calls[0].parsed is None


def test_answer_inside_bounds_and_string_numbers_are_accepted(models):
    m = models["nk3"]
    llm = mock_llm(default_response={"inflation_expectation_change_pp": "+2.0 percentage points"})
    spec = llm_spec()
    spec.llm.answer_bounds = {"inflation_expectation_change_pp": (-50.0, 50.0)}
    assert build_shock_path(spec, m, T, llm=llm).values[0] == pytest.approx(0.02)
