"""Experiment grid and runner (offline, mock LLM).

Expect:
- cells = models x scenarios x (LLMs for LLM scenarios) x intensities x modes x repeats; repeats only for LLMs with
  temperature != 0; shocks of other models are dropped and cells left without shocks are skipped
- each run directory has config, manifest, llm_calls, shock paths, trajectories (baseline/shocked/irf) and metrics
- a failing cell is recorded as failed and does not stop the sweep
- a linear model's response scales with intensity; baselines are solved once per model/solver/horizon
- no secret value ever reaches the outputs
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from dgse_llm.experiments import expand_cells, load_run, resolve_experiment, run_cell, run_experiment
from dgse_llm.llms import PromptRenderer
from tests.helpers import mock_llms, nk_experiment


def test_cell_expansion_counts_and_skipping():
    cells = expand_cells(nk_experiment())
    assert [(c.model.name, c.scenario.id) for c in cells] == [("nk3", "manual"), ("nk3", "wedge"), ("rbc", "manual")]
    assert [s.id for s in cells[0].scenario.shocks] == ["cp"], "rbc shock dropped for nk3"
    assert len({c.run_id for c in cells}) == 3


def test_sweep_multiplies_cells():
    assert len(expand_cells(nk_experiment(intensities=[0.001, 0.002]))) == 6
    assert len(expand_cells(nk_experiment(repeats=3, wedge_only=True, temperature=0.7))) == 3


def test_temperature_zero_runs_once_whatever_the_repeats():
    assert len(expand_cells(nk_experiment(repeats=3, wedge_only=True))) == 1


def test_llm_scenario_without_llms_is_an_error():
    cfg = nk_experiment().model_copy(update={"llms": []})
    with pytest.raises(ValueError, match="no llms"):
        expand_cells(cfg)


def test_end_to_end_writes_all_files(tmp_path):
    manifests = run_experiment(nk_experiment(), out_root=tmp_path)
    assert [m.status for m in manifests] == ["ok"] * 3
    exp = tmp_path / "exp"
    index = pd.read_parquet(exp / "index.parquet")
    assert len(index) == 3 and set(index["status"]) == {"ok"}
    for m in manifests:
        d = exp / m.run_id
        for name in ("config.resolved.yaml", "manifest.json", "llm_calls.jsonl", "shock_paths.parquet",
                     "trajectories.parquet", "metrics.json"):
            assert (d / name).exists(), f"{m.run_id}: {name}"
        run = load_run(d)
        assert set(run.trajectories["scenario"]) == {"baseline", "shocked", "irf"}
        assert list(run.trajectories.columns) == ["scenario", "variable", "t", "value"]
        assert run.trajectories["t"].max() == 29
        assert run.manifest.diagnostics["shocked"].converged
        assert run.manifest.modified and run.manifest.config_hash
        assert set(run.metrics) <= {"pi", "x", "i", "Y", "C"} and run.metrics


def test_manual_run_records_modified_variable_and_response(tmp_path):
    run = load_run(tmp_path / "exp" / next(m.run_id for m in run_experiment(nk_experiment(), out_root=tmp_path)
                                            if m.scenario == "manual" and m.model == "nk3"))
    assert run.manifest.modified == [{"variable": "u", "mode": "additive", "source": "manual", "shock_id": "cp"}]
    assert run.manifest.n_llm_calls == 0 and run.llm_calls == []
    assert run.metrics["pi"]["peak"] > 0, "cost-push raises inflation"


def test_llm_run_records_calls_and_paths(tmp_path):
    manifests = run_experiment(nk_experiment(), out_root=tmp_path)
    m = next(x for x in manifests if x.scenario == "wedge")
    run = load_run(tmp_path / "exp" / m.run_id)
    assert m.n_llm_calls == 2 and len(run.llm_calls) == 2
    assert all(c.parsed and c.prompt and not c.error for c in run.llm_calls)
    paths = run.shock_paths
    assert set(paths["variable"]) == {"wedge_pi"} and len(paths) == 30
    assert paths["value"].iloc[1:6].round(6).eq(0.005).all()
    assert paths["value"].iloc[6:].abs().max() == 0
    assert run.manifest.modified[0]["source"] == "llm"


def test_response_scales_linearly_with_intensity(tmp_path):
    cfg = nk_experiment(wedge_only=True, intensities=[1.0, 2.0])
    manifests = run_experiment(cfg, out_root=tmp_path)
    peaks = {m.intensity: load_run(tmp_path / "exp" / m.run_id).metrics["pi"]["peak"] for m in manifests}
    assert peaks[2.0] == pytest.approx(2 * peaks[1.0], rel=1e-6)


def test_failing_cell_is_recorded_and_sweep_continues(tmp_path):
    manifests = run_experiment(nk_experiment(with_bad=True), out_root=tmp_path)
    by_status = {s: [m for m in manifests if m.status == s] for s in ("ok", "failed")}
    assert len(by_status["ok"]) == 3 and len(by_status["failed"]) == 1
    bad = by_status["failed"][0]
    assert "not shockable" in bad.error
    d = tmp_path / "exp" / bad.run_id
    assert (d / "manifest.json").exists() and not (d / "trajectories.parquet").exists()
    index = pd.read_parquet(tmp_path / "exp" / "index.parquet")
    assert set(index["status"]) == {"ok", "failed"}


def test_baseline_solved_once_per_model(tmp_path):
    cfg = nk_experiment()
    baselines, llms = {}, {}
    for cell in [c for c in expand_cells(cfg) if c.model.name == "nk3"]:
        run_cell(cfg, cell, tmp_path, PromptRenderer(), llms, baselines)
    assert len(baselines) == 1 and len(llms) == 1


def test_repeats_get_distinct_runs(tmp_path):
    cfg = nk_experiment(wedge_only=True, repeats=2, temperature=0.7)
    manifests = run_experiment(cfg, out_root=tmp_path)
    assert sorted(m.repeat for m in manifests) == [0, 1]
    assert len({m.run_id for m in manifests}) == 2


def test_config_hash_is_deterministic(tmp_path):
    a = run_experiment(nk_experiment(), out_root=tmp_path / "a")
    b = run_experiment(nk_experiment(), out_root=tmp_path / "b")
    assert [m.config_hash for m in a] == [m.config_hash for m in b]


def test_secret_values_never_reach_outputs(tmp_path, monkeypatch):
    secret = "sk-test-0123456789abcdefghijklmnop"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    run_experiment(nk_experiment(api_key_env="OPENAI_API_KEY"), out_root=tmp_path)
    for f in Path(tmp_path).rglob("*"):
        if f.is_file() and f.suffix in {".yaml", ".json", ".jsonl", ".md"}:
            assert secret not in f.read_text(encoding="utf-8"), f
    cfg = json.loads(next((tmp_path / "exp").glob("*wedge*/manifest.json")).read_text(encoding="utf-8"))
    assert cfg["llm"] == "mock"


def small_spec(llms=None, T=15, **over):
    raw = {
        "id": "re", "model": "nk3", "llms": llms or mock_llms(), "T": T, "prompts": ["demand", "cost_push"],
        "personas": {"p": "A tester."}, "include_model_info": [False],
        "scenarios": {"one": {"shocks": {4: "energy_price_spike"}}, "wave": {"shocks": {4: "energy_price_spike", 8: "energy_price_fall"}}},
    }
    raw.update(over)
    return resolve_experiment(raw)


def test_rerun_keeps_completed_runs_untouched(tmp_path):
    cfg, skipped = small_spec(), []
    first = run_experiment(cfg, out_root=tmp_path)
    manifest_files = sorted((tmp_path / "re").glob("*/manifest.json"))
    before = {p: p.read_text(encoding="utf-8") for p in manifest_files}
    second = run_experiment(cfg, out_root=tmp_path, on_skip=skipped.append)
    assert len(first) == len(second) == len(skipped) == 4
    assert {p: p.read_text(encoding="utf-8") for p in manifest_files} == before, "files are not rewritten"
    assert [m.answer for m in second] == [m.answer for m in first]
    assert (tmp_path / "re" / "index.parquet").exists() and (tmp_path / "re" / "summary.parquet").exists()


def test_force_runs_everything_again(tmp_path):
    cfg, skipped = small_spec(), []
    run_experiment(cfg, out_root=tmp_path)
    run_experiment(cfg, out_root=tmp_path, on_skip=skipped.append, force=True)
    assert skipped == []


def test_a_changed_setup_is_run_again_and_the_rest_is_kept(tmp_path):
    run_experiment(small_spec(), out_root=tmp_path)
    skipped = []
    run_experiment(small_spec(T=20), out_root=tmp_path, on_skip=skipped.append)
    assert skipped == [], "the horizon is part of every run's setup"
    skipped = []
    more = small_spec(T=20, scenarios={
        "one": {"shocks": {4: "energy_price_spike"}},
        "three": {"shocks": {4: "energy_price_spike", 8: "energy_price_fall", 12: "credit_crunch"}},
    })
    manifests = run_experiment(more, out_root=tmp_path, on_skip=skipped.append)
    assert len(manifests) == 4 and len(skipped) == 2, "the unchanged 'one' scenario is kept for both prompts"


def test_an_edited_prompt_template_is_run_again(tmp_path):
    import shutil

    from dgse_llm.paths import prompts_dir

    copy = tmp_path / "prompts"
    shutil.copytree(prompts_dir(), copy)
    cfg, skipped = small_spec(), []
    run_experiment(cfg, out_root=tmp_path / "out", renderer=PromptRenderer(copy))
    demand = copy / "nk3" / "demand.j2"
    demand.write_text(demand.read_text(encoding="utf-8") + "\nBe concise.\n", encoding="utf-8")
    run_experiment(cfg, out_root=tmp_path / "out", renderer=PromptRenderer(copy), on_skip=skipped.append)
    assert sorted(m.meta["prompt"] for m in skipped) == ["cost_push", "cost_push"], "only the demand cells re-ran"


def wild_llm(value):
    return {"wild": {"provider": "mock", "temperature": 0.0, "cache": False,
                     "options": {"default_response": {"change_pct": value}}}}


def test_failed_runs_are_logged_by_error_name_with_the_returned_text(tmp_path):
    manifests = run_experiment(small_spec(llms=wild_llm(9000)), out_root=tmp_path)
    assert {m.status for m in manifests} == {"failed"}
    logs = sorted((tmp_path / "re" / "errors").glob("*/*.json"))
    assert len(logs) == 4 and {p.parent.name for p in logs} == {"ParseError"}
    record = json.loads(logs[0].read_text(encoding="utf-8"))
    assert record["error_type"] == "ParseError" and "less than or equal to" in record["error"]
    call = record["llm_calls"][0]
    assert "9000" in call["response_text"] and "Estimate the percentage change" in call["prompt"]
    assert record["axes"]["persona"] == "p" and record["run_id"] == logs[0].stem


def test_other_failures_get_their_own_error_folder(tmp_path):
    run_experiment(nk_experiment(with_bad=True), out_root=tmp_path)
    logs = list((tmp_path / "exp" / "errors").glob("*/*.json"))
    assert len(logs) == 1 and logs[0].parent.name == "ShockError"
    assert "not shockable" in json.loads(logs[0].read_text(encoding="utf-8"))["error"]


def test_failed_runs_are_retried_and_their_error_log_cleared_on_success(tmp_path):
    run_experiment(small_spec(llms=wild_llm(9000)), out_root=tmp_path)
    skipped = []
    manifests = run_experiment(small_spec(llms=wild_llm(1.0)), out_root=tmp_path, on_skip=skipped.append)
    assert skipped == [] and {m.status for m in manifests} == {"ok"}
    assert list((tmp_path / "re" / "errors").glob("*/*.json")) == []


def test_a_failed_rerun_leaves_no_stale_results(tmp_path):
    run_experiment(small_spec(llms=wild_llm(1.0)), out_root=tmp_path)
    run = next((tmp_path / "re").glob("*demand.p.one*"))
    assert (run / "trajectories.parquet").exists()
    run_experiment(small_spec(llms=wild_llm(9000)), out_root=tmp_path)
    assert not (run / "trajectories.parquet").exists() and not (run / "metrics.json").exists()
    assert json.loads((run / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.slow
def test_krusell_smith_household_consumption_experiment(tmp_path):
    cfg = resolve_experiment({
        "id": "ks", "model": "krusell_smith", "llms": mock_llms(), "T": 20,
        "prompts": ["household_consumption"], "personas": ["normal"],
        "scenarios": {"pandemic": {"shocks": {4: "pandemic_rumours"}}}, "include_model_info": [False],
    })
    manifests = run_experiment(cfg, out_root=tmp_path)
    assert [m.status for m in manifests] == ["ok"], [m.error for m in manifests]
    run = load_run(tmp_path / "ks" / manifests[0].run_id)
    assert manifests[0].answer == -15.0 and manifests[0].n_llm_calls == 1
    assert run.metrics["C"]["peak"] < 0, "a positive taste shifter lowers consumption"
    assert run.metrics["d"]["peak"] == pytest.approx(0.015)
