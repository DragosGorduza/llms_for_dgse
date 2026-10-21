"""Experiment files compile into runs.

Expect: personas, narratives and shock texts resolve from experiments/; a scenario maps periods to shock names (several
periods, ranges, lists, the same shock twice); the grid is the inner product
prompts x personas x scenarios x include_model_info (x LLMs, x repeats only when temperature != 0); each cell is one
LLM shock with the right input, event periods, conversion, bounds and prompt variables; model info is in the prompt
only when the flag is on; shock files carry no timing; periods are validated against T.
"""

import pytest

from dgse_llm.experiments import expand_cells, load_experiment, resolve_experiment, resolve_spec
from dgse_llm.experiments.loader import load_shock_library
from tests.helpers import mock_llms


def spec(**over):
    raw = {
        "id": "x", "model": "nk3", "llms": mock_llms(), "T": 20,
        "prompts": ["demand", "cost_push"],
        "personas": {"p": "PERSONA P"},
        "scenarios": {
            "one": {"shocks": {4: "energy_price_spike"}},
            "wave": {"narrative": "recession", "shocks": {4: "energy_price_spike", 12: "energy_price_fall"}},
        },
    }
    raw.update(over)
    return raw


def test_names_resolve_to_libraries():
    s = resolve_spec(spec(personas="all"))
    assert s.model.name == "nk3" and list(s.llms) == ["mock"]
    assert {"normal", "economist"} <= set(s.personas)
    assert s.scenarios["wave"].narrative.startswith("The economy is already in a recession")
    assert s.scenarios["one"].narrative == ""
    assert [o.periods for o in s.scenarios["wave"].events] == [[4], [12]]


def test_literal_narrative_text_is_kept():
    s = resolve_spec(spec(scenarios={"one": {"narrative": "A custom context.", "shocks": {4: "energy_price_spike"}}}))
    assert s.scenarios["one"].narrative == "A custom context."


def test_unknown_shock_or_persona_is_an_error():
    with pytest.raises(KeyError, match="unknown shock"):
        resolve_spec(spec(scenarios={"s": {"shocks": {4: "nope"}}}))
    with pytest.raises(KeyError, match="Unknown persona"):
        resolve_spec(spec(personas=["nobody"]))


def test_a_scenario_needs_shocks_mapped_to_periods():
    with pytest.raises(ValueError, match="map periods"):
        resolve_spec(spec(scenarios={"s": {"narrative": "recession"}}))
    with pytest.raises(ValueError, match="map periods"):
        resolve_spec(spec(scenarios={"s": {"shocks": ["energy_price_spike"]}}))
    with pytest.raises(ValueError, match="map periods"):
        resolve_spec(spec(scenarios={"s": {"shocks": {}}}))


def test_different_shocks_at_different_periods_in_any_order():
    s = resolve_spec(spec(scenarios={"mix": {"shocks": {
        20: "technology_breakthrough", 5: "energy_price_spike", 12: "sovereign_default",
    }}}))
    events = s.scenarios["mix"].events
    assert [(o.periods, o.name) for o in events] == [
        ([5], "energy_price_spike"), ([12], "sovereign_default"), ([20], "technology_breakthrough"),
    ]
    assert events[0].text.startswith("Energy prices have jumped")
    assert s.scenarios["mix"].shocks == ["energy_price_spike", "sovereign_default", "technology_breakthrough"]


def test_the_same_shock_can_happen_twice_and_lists_put_several_in_one_period():
    s = resolve_spec(spec(scenarios={
        "twice": {"shocks": {5: "energy_price_spike", 12: "energy_price_spike"}},
        "together": {"shocks": {4: ["energy_price_spike", "bank_collapse"], 8: "credit_crunch"}},
    }))
    assert [o.periods for o in s.scenarios["twice"].events] == [[5], [12]]
    assert [(o.periods, o.name) for o in s.scenarios["together"].events] == [
        ([4], "energy_price_spike"), ([4], "bank_collapse"), ([8], "credit_crunch"),
    ]


def test_period_ranges_and_backwards_ranges():
    s = resolve_spec(spec(scenarios={"r": {"shocks": {"4-12": "energy_price_spike", 3: "energy_price_fall", "5": "credit_crunch"}}}))
    assert [o.periods for o in s.scenarios["r"].events] == [[3], [5], list(range(4, 13))]
    with pytest.raises(ValueError, match="backwards"):
        resolve_spec(spec(scenarios={"r": {"shocks": {"9-4": "credit_crunch"}}}))


def test_bare_model_name_and_mapping_forms():
    assert resolve_spec(spec(model="rbc")).model.params == {}
    s = resolve_spec(spec(model={"name": "rbc", "params": {"alpha": 0.5}, "solver": "numpy_newton"}))
    assert s.model.params == {"alpha": 0.5}


def test_grid_is_the_inner_product_and_model_info_defaults_to_both():
    cfg = resolve_experiment(spec())
    assert len(cfg.scenarios) == 2 * 1 * 2 * 2
    assert len(expand_cells(cfg)) == len(cfg.scenarios)
    assert len({s.id for s in cfg.scenarios}) == len(cfg.scenarios)
    assert {s.meta["model_info"] for s in cfg.scenarios} == {False, True}


def test_each_scenario_is_one_llm_shock_with_the_right_wiring():
    cfg = resolve_experiment(spec(include_model_info=[False]))
    by_id = {s.id: s for s in cfg.scenarios}
    s = by_id["cost_push.p.wave.noinfo"]
    shock = s.shocks[0]
    assert s.meta == {"prompt": "cost_push", "persona": "p", "scenario": "wave", "model_info": False}
    assert shock.source == "llm" and shock.variable == "u" and shock.model == "nk3"
    assert [o.periods for o in shock.llm.occurrences] == [[4], [12]]
    assert shock.llm.occurrences[0].text.startswith("Energy prices have jumped")
    assert shock.llm.template == "nk3/cost_push.j2" and shock.llm.keys == ["change_pct"]
    assert shock.llm.mapping == "linear" and shock.llm.mapping_params == {"slope": 0.0002}
    assert not shock.llm.scale_by_baseline and shock.llm.narrative.startswith("The economy is already")
    assert shock.llm.answer_bounds == {"change_pct": (-100.0, 100.0)}
    single = by_id["cost_push.p.one.noinfo"].shocks[0].llm
    assert [o.periods for o in single.occurrences] == [[4]] and single.narrative == ""


def test_baseline_relative_prompts_scale_by_the_inputs_own_level():
    cfg = resolve_experiment(spec(model="rbc", prompts=["tfp"], include_model_info=[False]))
    llm = cfg.scenarios[0].shocks[0].llm
    assert llm.scale_by_baseline and llm.mapping == "pct" and llm.mapping_params == {}


def test_model_info_is_only_in_template_vars_when_the_flag_is_on():
    cfg = resolve_experiment(spec(model="rbc", prompts=["tfp"]))
    by_flag = {s.meta["model_info"]: s.shocks[0].llm.template_vars for s in cfg.scenarios}
    assert by_flag[False]["model_info"] == ""
    assert "Real business cycle" in by_flag[True]["model_info"] and "alpha = 0.33" in by_flag[True]["model_info"]
    assert by_flag[True]["persona"] == "PERSONA P"


def test_parameter_overrides_show_up_in_model_info():
    cfg = resolve_experiment(spec(model={"name": "rbc", "params": {"alpha": 0.5}}, prompts=["tfp"], include_model_info=[True]))
    assert "alpha = 0.5" in cfg.scenarios[0].shocks[0].llm.template_vars["model_info"]


def test_unknown_prompt_for_a_model_is_an_error():
    with pytest.raises(KeyError, match="no prompt"):
        resolve_experiment(spec(prompts=["tfp"]))


def test_event_periods_must_fit_the_horizon():
    with pytest.raises(ValueError, match="outside"):
        resolve_experiment(spec(T=10))


def test_shock_files_carry_no_timing(tmp_path):
    (tmp_path / "shocks").mkdir()
    (tmp_path / "shocks" / "ok.yaml").write_text("text: Something happens.\ndescription: note\n", encoding="utf-8")
    assert load_shock_library(tmp_path) == {"ok": "Something happens."}
    (tmp_path / "shocks" / "timed.yaml").write_text("text: Late.\nperiods: [4]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected keys"):
        load_shock_library(tmp_path)
    (tmp_path / "shocks" / "timed.yaml").write_text("description: no text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text"):
        load_shock_library(tmp_path)


def test_llms_mapping_runs_every_entry_and_repeats_only_sampled_ones():
    hot = {"provider": "mock", "temperature": 0.7, "cache": False}
    reasoner = {"provider": "mock", "temperature": None, "cache": False}
    cfg = resolve_experiment(spec(
        llms={**mock_llms(), "hot": hot, "reasoner": reasoner}, repeats=3, include_model_info=[False],
        prompts=["demand"],
    ))
    assert [llm.name for llm in cfg.llms] == ["mock", "hot", "reasoner"]
    cells = expand_cells(cfg)
    per_llm = {name: [c for c in cells if c.llm.name == name] for name in ("mock", "hot", "reasoner")}
    assert [len(v) for v in per_llm.values()] == [2, 6, 6]  # 2 scenarios; 3 draws for temperature != 0
    assert sorted({c.repeat for c in per_llm["hot"]}) == [0, 1, 2]
    assert len({c.run_id for c in cells}) == len(cells)


def test_load_experiment_from_file(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_text(
        "id: e\nmodel: rbc\nT: 10\nprompts: [tfp]\npersonas: {p: who}\n"
        "llms:\n  m: {provider: mock, temperature: 0.0, api_key_env: null}\n"
        "scenarios:\n  s: {shocks: {4: pandemic_rumours}}\ninclude_model_info: [true]\n",
        encoding="utf-8",
    )
    cfg = load_experiment(f)
    assert cfg.id == "e" and len(cfg.scenarios) == 1 and cfg.scenarios[0].meta["model_info"] is True
    assert cfg.llms[0].name == "m" and cfg.llms[0].api_key_env is None
