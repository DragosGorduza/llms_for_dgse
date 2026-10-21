"""Like-with-like evaluation of an experiment run.

Expect: pairs are formed only inside cells that share every other axis, model info is ordered without -> with and
differences are b - a, repeats are averaged with a z score, axis importance measures the range across levels, and a
mock nk3 experiment with two LLMs, two personas and both model-info settings writes evaluations/ with tables and
figures; the CLI can evaluate a run by path or right after `run --evaluate`.
"""

import pandas as pd
import pytest

from dgse_llm.cli import main
from dgse_llm.evaluation import (
    axis_importance, cell_stats, evaluate_experiment, pair_table, summarize_pairs, trajectory_pairs,
)
from dgse_llm.evaluation.compare import irf_cells, ordered_levels
from dgse_llm.experiments import resolve_experiment, run_experiment
from dgse_llm.schemas import dump_yaml
from tests.helpers import MOCK_RULES, mock_llms

BASE = dict(model="nk3", llm="m", prompt="demand", persona="p", scenario="one", model_info="without",
            variable="x", metric="peak")


def long(rows):
    return pd.DataFrame([{**BASE, **r} for r in rows])


def test_model_info_is_ordered_without_then_with():
    assert ordered_levels("model_info", ["with", "without"]) == ["without", "with"]
    assert ordered_levels("persona", ["b", "a"]) == ["a", "b"]


def test_pairs_only_inside_like_for_like_cells():
    cells = cell_stats(long([
        {"model_info": "without", "value": 1.0}, {"model_info": "with", "value": 4.0},
        {"model_info": "without", "value": 2.0, "persona": "q"}, {"model_info": "with", "value": 1.0, "persona": "q"},
        {"model_info": "with", "value": 9.0, "persona": "lonely"},
    ]))
    pairs = pair_table(cells, "model_info").set_index("persona")
    assert set(pairs.index) == {"p", "q"}, "a cell with one level has no pair"
    assert (pairs["a"] == "without").all() and (pairs["b"] == "with").all()
    assert pairs.loc["p", "diff"] == pytest.approx(3.0) and pairs.loc["q", "diff"] == pytest.approx(-1.0)
    s = summarize_pairs(pair_table(cells, "model_info")).iloc[0]
    assert s["n_cells"] == 2 and s["mean_diff"] == pytest.approx(1.0) and s["mean_abs_diff"] == pytest.approx(2.0)
    assert s["share_b_higher"] == pytest.approx(0.5)


def test_repeats_are_averaged_and_give_a_z_score():
    cells = cell_stats(long([
        {"llm": "a", "value": 1.0}, {"llm": "a", "value": 3.0},
        {"llm": "b", "value": 5.0}, {"llm": "b", "value": 7.0},
    ]))
    pair = pair_table(cells, "llm").iloc[0]
    assert pair["a_mean"] == 2.0 and pair["b_mean"] == 6.0 and pair["a_n"] == 2
    assert pair["z"] == pytest.approx(4.0 / 2 ** 0.5)


def test_axis_importance_is_the_range_across_levels():
    cells = cell_stats(long([
        {"persona": "a", "value": 1.0}, {"persona": "b", "value": 3.0}, {"persona": "c", "value": 2.0},
    ]))
    imp = axis_importance(cells, ["persona", "llm"])
    assert list(imp["axis"]) == ["persona"], "llm has one level, so no row"
    assert imp.iloc[0]["mean_range"] == pytest.approx(2.0)
    assert imp.iloc[0]["mean_rel_range"] == pytest.approx(1.0)


def test_trajectory_pairs_measure_path_distance():
    irf = pd.DataFrame([
        {**BASE, "llm": llm, "t": t, "value": v}
        for llm, path in (("a", [0.0, 1.0, 0.5]), ("b", [0.0, 2.0, 1.0]))
        for t, v in enumerate(path)
    ]).drop(columns="metric")
    pairs, diffs = trajectory_pairs(irf_cells(irf), "llm")
    row = pairs.iloc[0]
    assert row["a"] == "a" and row["b"] == "b" and row["max_abs_diff"] == pytest.approx(1.0)
    assert row["max_abs_diff_t"] == 1 and row["corr"] == pytest.approx(1.0) and row["cum_diff"] == pytest.approx(1.5)
    assert diffs["diff"].tolist() == pytest.approx([0.0, 1.0, 0.5])


def spec():
    other = {"provider": "mock", "temperature": 0.0, "cache": False,
             "options": {"rules": [{**r, "response": {"change_pct": r["response"]["change_pct"] / 2}}
                                   for r in MOCK_RULES], "default_response": {"change_pct": 0.0}}}
    return {
        "id": "eval", "model": "nk3", "llms": {**mock_llms(), "half": other}, "T": 20,
        "prompts": ["demand"],
        "personas": {"normal": "A typical household.", "economist": "An economist."},
        "scenarios": {"one": {"shocks": {4: "energy_price_spike"}}},
        "tracked": ["x", "pi"],
    }


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    manifests = run_experiment(resolve_experiment(spec()), out_root=root)
    assert all(m.status == "ok" for m in manifests), [m.error for m in manifests]
    return evaluate_experiment(root / "eval")


def test_evaluation_writes_report_tables_and_figures(evaluated):
    out = evaluated.parent
    assert out.name == "evaluations"
    text = evaluated.read_text(encoding="utf-8")
    for heading in ("## Which axis moves results most", "(`model_info`)", "(`llm`)", "(`persona`)"):
        assert heading in text, heading
    assert "Skipped (one level only): `scenario`" in text
    for axis in ("model_info", "llm", "persona"):
        assert (out / axis / "pairs.parquet").exists() and (out / axis / "trajectory_pairs.parquet").exists()
        assert list((out / axis / "figures").glob("cell__*.png")), axis
        assert list((out / axis / "figures").glob("scatter_peak__*.png")), axis


def test_llm_comparison_sees_the_halved_answer(evaluated):
    pairs = pd.read_parquet(evaluated.parent / "llm" / "pairs.parquet")
    answers = pairs[pairs["variable"] == "llm_answer"]
    assert (answers["a"] == "half").all() and (answers["b"] == "mock").all()
    assert answers["a_mean"].tolist() == pytest.approx([-2.5] * len(answers))
    assert answers["diff"].tolist() == pytest.approx([-2.5] * len(answers))


def test_evaluation_can_be_rerun(evaluated):
    assert evaluate_experiment(evaluated.parent.parent) == evaluated


def test_cli_evaluate_and_run_with_evaluate(tmp_path):
    raw = {**spec(), "id": "cli_eval", "llms": mock_llms(), "personas": {"p": "A tester."}}
    path = tmp_path / "cli_eval.yaml"
    dump_yaml(raw, path)
    assert main(["run", str(path), "--out", str(tmp_path / "out"), "--evaluate"]) == 0
    exp = tmp_path / "out" / "cli_eval"
    assert (exp / "evaluations" / "evaluation.md").exists()
    assert main(["evaluate", str(exp), "--axes", "model_info"]) == 0
