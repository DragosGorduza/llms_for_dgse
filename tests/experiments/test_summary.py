"""Distribution of answers across repeats and the effect of showing the model to the LLM.

Expect: at temperature 0 one draw per cell (std 0); at temperature > 0 `repeats` draws whose mean, spread and range
are recorded; with and without model info are paired per prompt/persona/scenario/LLM; a full mock experiment
writes index, summary and info_effect files and the experiment report shows both tables.
"""

import pandas as pd
import pytest

from dgse_llm.experiments import (
    build_summary, info_effect, load_run, resolve_experiment, run_experiment, write_summary,
)
from dgse_llm.reporting import write_experiment_report
from tests.helpers import mock_llms

BASE = dict(model="nk3", llm="m", prompt="demand", persona="p", scenario="one")


def runs(rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def test_summary_describes_the_draws_of_each_cell():
    df = runs([
        {"model_info": False, "answer": 1.0, "peak_x": 0.1},
        {"model_info": False, "answer": 3.0, "peak_x": 0.3},
        {"model_info": False, "answer": 5.0, "peak_x": 0.5},
        {"model_info": True, "answer": 10.0, "peak_x": 1.0},
    ])
    s = build_summary(df).set_index("model_info")
    off, on = s.loc[False], s.loc[True]
    assert off["n"] == 3 and off["answer_mean"] == pytest.approx(3.0) and off["answer_median"] == 3.0
    assert off["answer_min"] == 1.0 and off["answer_max"] == 5.0 and off["answer_std"] == pytest.approx(2.0)
    assert off["peak_x_mean"] == pytest.approx(0.3)
    assert on["n"] == 1 and on["answer_std"] == 0.0 and on["peak_x_std"] == 0.0


def test_info_effect_pairs_with_and_without():
    s = build_summary(runs([
        {"model_info": False, "answer": 2.0}, {"model_info": True, "answer": 5.0},
        {"model_info": False, "answer": 4.0, "scenario": "two"}, {"model_info": True, "answer": 1.0, "scenario": "two"},
    ]))
    effect = info_effect(s).set_index("scenario")
    assert effect.loc["one", "diff"] == pytest.approx(3.0) and effect.loc["two", "diff"] == pytest.approx(-3.0)
    assert effect.loc["one", "answer_without"] == 2.0 and effect.loc["one", "answer_with"] == 5.0


def test_info_effect_is_empty_unless_both_settings_ran():
    assert info_effect(build_summary(runs([{"model_info": False, "answer": 1.0}]))).empty
    assert build_summary(pd.DataFrame()).empty


def spec(**over):
    raw = {
        "id": "sum", "model": "nk3", "llms": mock_llms(), "T": 20,
        "prompts": ["demand", "inflation_expectations"],
        "personas": {"normal": "A typical household."},
        "scenarios": {
            "one": {"shocks": {4: "energy_price_spike"}},
            "wave": {"shocks": {4: "energy_price_spike", 12: "energy_price_fall"}},
        },
        "tracked": ["x", "pi", "wedge_pi"],
    }
    raw.update(over)
    return raw


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    manifests = run_experiment(resolve_experiment(spec()), out_root=root)
    return root / "sum", manifests


def test_mock_experiment_runs_every_cell_and_records_answers(outputs):
    exp, manifests = outputs
    assert len(manifests) == 2 * 1 * 1 * 2 * 2 and all(m.status == "ok" for m in manifests), [m.error for m in manifests]
    by_prompt = {m.meta["prompt"]: m.answer for m in manifests}
    assert by_prompt == {"demand": -5.0, "inflation_expectations": 50.0}
    for name in ("index.parquet", "summary.parquet", "info_effect.parquet"):
        assert (exp / name).exists(), name
    index = pd.read_parquet(exp / "index.parquet")
    assert {"prompt", "persona", "scenario", "model_info", "answer"} <= set(index.columns)


def test_shock_lands_in_the_event_periods_in_model_units(outputs):
    exp, manifests = outputs
    wave = next(m for m in manifests if m.meta["prompt"] == "inflation_expectations" and m.meta["scenario"] == "wave")
    paths = load_run(exp / wave.run_id).shock_paths
    assert paths[paths["active"]]["t"].tolist() == [4, 12]
    assert paths[paths["active"]]["value"].round(8).tolist() == [0.0025, 0.0025]  # 50% x 0.00005 per 1%
    assert wave.n_llm_calls == 2, "one call per event"
    one = next(m for m in manifests if m.meta["prompt"] == "demand" and m.meta["scenario"] == "one")
    assert load_run(exp / one.run_id).shock_paths.query("active")[["t", "value"]].round(8).values.tolist() == [[4, -0.05]]


def test_model_info_flag_changes_the_prompt_not_the_run_count(outputs):
    exp, manifests = outputs
    prompts = {}
    for m in manifests:
        if m.meta["prompt"] == "demand" and m.meta["scenario"] == "one":
            prompts[m.meta["model_info"]] = load_run(exp / m.run_id).llm_calls[0].prompt
    assert "Three-equation" not in prompts[False] and "Three-equation" in prompts[True]
    assert "kappa = 0.1" in prompts[True] and "kappa" not in prompts[False]
    assert "Energy prices have jumped" in prompts[False]


def test_experiment_report_has_distribution_and_info_effect(outputs):
    exp, _ = outputs
    text = write_experiment_report(exp).read_text(encoding="utf-8")
    assert "## Answers by cell (relative % change)" in text
    assert "## Effect of showing the model to the LLM" in text


def test_write_summary_can_be_rerun(outputs):
    exp, _ = outputs
    assert not write_summary(exp).empty


def test_sampled_llm_records_a_distribution(tmp_path):
    hot = {"hot": {"provider": "mock", "temperature": 0.7, "cache": False,
                    "options": {"default_response": {"change_pct": 2.0}}}}
    cfg = resolve_experiment(spec(llms=hot, repeats=3, prompts=["demand"], scenarios={"one": {"shocks": {4: "energy_price_spike"}}},
                                  include_model_info=[False]))
    manifests = run_experiment(cfg, out_root=tmp_path)
    assert sorted(m.repeat for m in manifests) == [0, 1, 2]
    summary = pd.read_parquet(tmp_path / "sum" / "summary.parquet")
    assert len(summary) == 1 and summary.loc[0, "n"] == 3 and summary.loc[0, "answer_mean"] == 2.0
