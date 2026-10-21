"""Markdown reports.

Expect: a run report has setup, modified variables, LLM prompts and answers, impulse-response figure, metrics and
diagnostics; a failed run report shows the error; the experiment report lists every run and the failures.
"""

import pandas as pd
import pytest

from dgse_llm.reporting import md_table, write_experiment_report, write_run_report
from dgse_llm.reporting import report as report_module
from dgse_llm.experiments import run_experiment
from tests.helpers import nk_arms_experiment, nk_experiment


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    manifests = run_experiment(nk_experiment(with_bad=True), out_root=root)
    return root / "exp", manifests


def _run_dir(outputs, scenario, status="ok"):
    exp, manifests = outputs
    m = next(x for x in manifests if x.scenario == scenario and x.status == status)
    return exp / m.run_id


def test_llm_run_report_sections(outputs):
    path = write_run_report(_run_dir(outputs, "wedge"))
    text = path.read_text(encoding="utf-8")
    for heading in ("## Setup", "## Modified variables", "## LLM calls (2)", "## Model paths", "## Impulse responses",
                    "## Metrics", "## Solver diagnostics", "## Provenance"):
        assert heading in text, heading
    assert "Energy prices have jumped 30% overnight." in text
    assert "inflation_expectation_change_pp" in text
    assert "wedge_pi" in text
    assert (path.parent / "figures" / "irf.png").exists()
    assert (path.parent / "figures" / "trajectories.png").exists()
    assert "all scenario shocks removed" in text
    assert (path.parent / "figures" / "shock_paths.png").exists()


def test_manual_run_report_has_no_llm_calls(outputs):
    text = write_run_report(_run_dir(outputs, "manual")).read_text(encoding="utf-8")
    assert "## LLM calls (0)" in text and "cp" in text


def test_failed_run_report_shows_error(outputs):
    text = write_run_report(_run_dir(outputs, "bad", "failed")).read_text(encoding="utf-8")
    assert "**failed**" in text and "not shockable" in text
    assert "## Impulse responses" not in text


def test_experiment_report_lists_all_runs_and_failures(outputs):
    exp, manifests = outputs
    text = write_experiment_report(exp).read_text(encoding="utf-8")
    for m in manifests:
        assert m.run_id in text
    assert "## Failures" in text and "## Peak response by run" in text
    assert (exp / "figures" / "irf_overlay.png").exists()


def test_four_economy_report_when_classic_and_llm_shocks_are_combined(tmp_path):
    manifest = run_experiment(nk_arms_experiment(), out_root=tmp_path)[0]
    path = write_run_report(tmp_path / "arms" / manifest.run_id)
    text = path.read_text(encoding="utf-8")
    assert "## Classic shock, LLM shock and both together" in text
    assert "### peak response" in text and "### cumulative response" in text
    assert "interaction" in text and "classic + LLM" in text
    assert (path.parent / "figures" / "trajectories.png").exists()


def test_single_kind_report_has_no_arm_section(outputs):
    text = write_run_report(_run_dir(outputs, "manual")).read_text(encoding="utf-8")
    assert "Classic shock, LLM shock and both together" not in text


def test_md_table_formats_values_and_escapes_pipes():
    out = md_table([{"a": 1.23456789, "b": None, "c": "x|y"}], ["a", "b", "c"])
    assert "1.235" in out and "| - |" in out and "x\\|y" in out
    assert md_table([], ["a"]) == "_none_\n"


@pytest.mark.parametrize("plot_name", ["_plot_trajectories", "_plot_irfs"])
def test_run_plots_mark_named_shock_onset(tmp_path, monkeypatch, plot_name):
    scenarios = ("baseline", "shocked", "irf")
    trajectories = pd.DataFrame([
        {"scenario": scenario, "variable": "Y", "t": period, "value": float(period)}
        for scenario in scenarios
        for period in range(4)
    ])
    shocks = pd.DataFrame([
        {"shock_id": "scenario-1", "variable": "wedge_pi", "mode": "additive", "t": period,
         "value": float(period > 0 and period < 3), "active": 0 < period < 3}
        for period in range(4)
    ])
    config = {"scenario": {"shocks": [{
        "id": "scenario-1", "variable": "wedge_pi", "source": "llm",
        "llm": {"occurrences": [{"periods": [1, 2], "name": "energy_price_spike"}]},
    }]}}
    figures = []
    monkeypatch.setattr(report_module.plt, "close", lambda figure: figures.append(figure))

    getattr(report_module, plot_name)(
        trajectories, tmp_path / "plot.png", "nk3", ["llm"], shocks, config
    )

    markers = [line for line in figures[-1].axes[0].lines if line.get_label() == "Shock: energy_price_spike (wedge_pi)"]
    assert len(markers) == 1
    assert list(markers[0].get_xdata()) == [1, 1]
    assert markers[0].get_linestyle() == "--"


@pytest.mark.parametrize("plot_name", ["_plot_trajectories", "_plot_irfs"])
def test_plots_show_orange_baseline(tmp_path, monkeypatch, plot_name):
    trajectories = pd.DataFrame([
        {"scenario": scenario, "variable": "Y", "t": period, "value": value}
        for scenario, values in (
            ("baseline", [2.0, 2.5]),
            ("shocked", [3.0, 4.0]),
            ("irf", [1.0, 1.5]),
        )
        for period, value in enumerate(values)
    ])
    figures = []
    close = report_module.plt.close
    monkeypatch.setattr(report_module.plt, "close", lambda figure: figures.append(figure))
    try:
        getattr(report_module, plot_name)(trajectories, tmp_path / "plot.png")
        axes = figures[-1].axes[0]
        lines = {line.get_label(): line for line in axes.lines}
        baseline = lines["base-model"]
        assert baseline.get_color() == "tab:orange"
        assert "base-model" in [text.get_text() for text in axes.get_legend().get_texts()]
        if plot_name == "_plot_trajectories":
            assert list(baseline.get_ydata()) == [2.0, 2.5]
            assert list(lines["scenario"].get_ydata()) == [3.0, 4.0]
        else:
            assert list(baseline.get_ydata()) == [0, 0]
            assert list(lines["scenario minus baseline"].get_ydata()) == [1.0, 1.5]
        assert (tmp_path / "plot.png").exists()
    finally:
        for figure in figures:
            close(figure)
