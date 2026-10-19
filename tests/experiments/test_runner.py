"""Experiment grid and runner (offline, mock LLM).

Expect:
- cells = models x scenarios x (LLMs for LLM scenarios) x intensities x modes x repeats; shocks of other models are dropped
  and cells left without shocks are skipped
- each run directory has config, manifest, llm_calls, shock paths, trajectories (baseline/shocked/irf) and metrics
- a failing cell is recorded as failed and does not stop the sweep
- a linear model's response scales with intensity; baselines are solved once per model/solver/horizon
- no secret value ever reaches the outputs
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from dgse_llm.experiments import expand_cells, load_experiment, load_run, run_cell, run_experiment
from dgse_llm.llms import PromptRenderer
from dgse_llm.paths import configs_dir
from tests.helpers import nk_experiment


def test_cell_expansion_counts_and_skipping():
    cells = expand_cells(nk_experiment())
    assert [(c.model.name, c.scenario.id) for c in cells] == [("nk3", "manual"), ("nk3", "wedge"), ("rbc", "manual")]
    assert [s.id for s in cells[0].scenario.shocks] == ["cp"], "rbc shock dropped for nk3"
    assert len({c.run_id for c in cells}) == 3


def test_sweep_multiplies_cells():
    assert len(expand_cells(nk_experiment(intensities=[0.001, 0.002]))) == 6
    assert len(expand_cells(nk_experiment(repeats=3, wedge_only=True))) == 3


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


@pytest.mark.slow
def test_krusell_smith_pandemic_experiment(tmp_path):
    cfg = load_experiment(configs_dir() / "experiments" / "ks_pandemic_mock.yaml").model_copy(update={"T": 40})
    manifests = run_experiment(cfg, out_root=tmp_path)
    assert [m.status for m in manifests] == ["ok"] * 3, [m.error for m in manifests]
    for m in manifests:
        run = load_run(tmp_path / cfg.id / m.run_id)
        assert m.n_llm_calls == 6
        assert run.metrics["C"]["peak"] < 0, "a positive taste shifter lowers consumption"
        assert run.metrics["d"]["peak"] > 0
