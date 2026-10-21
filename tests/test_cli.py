"""Command line interface.

Expect: `list` prints the registries, `run --report` executes a config and writes reports, `report` works on a
run directory and on an experiment directory, and a failing run gives a non-zero exit code.
"""

import pytest

from dgse_llm.cli import main
from dgse_llm.schemas import dump_yaml
from tests.helpers import mock_llms


def spec_file(tmp_path, **over):
    raw = {
        "id": "cli", "model": "rbc", "llms": mock_llms(), "T": 15, "prompts": ["tfp"],
        "personas": {"p": "A tester."}, "scenarios": {"one": {"shocks": {4: "pandemic_rumours"}}},
    }
    raw.update(over)
    path = tmp_path / "cli.yaml"
    dump_yaml(raw, path)
    return path


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for word in ("models:", "krusell_smith", "numpy_newton", "mock", "at_periods", "linear", "shockable inputs",
                 "prompts:", "tfp -> Z", "household_consumption -> d"):
        assert word in out


def test_run_and_report(tmp_path, capsys):
    cfg = spec_file(tmp_path)
    assert main(["run", str(cfg), "--out", str(tmp_path / "out"), "--report"]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok]") == 2, "with and without model info"
    exp = tmp_path / "out" / "cli"
    assert (exp / "experiment_report.md").exists()
    run_dirs = [d for d in exp.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    assert len(run_dirs) == 2 and all((d / "report.md").exists() for d in run_dirs)


def test_report_command_picks_report_type(tmp_path):
    cfg = spec_file(tmp_path)
    main(["run", str(cfg), "--out", str(tmp_path / "out")])
    exp = tmp_path / "out" / "cli"
    assert main(["report", str(exp)]) == 0 and (exp / "experiment_report.md").exists()
    run_dir = next(d for d in exp.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    assert main(["report", str(run_dir)]) == 0 and (run_dir / "report.md").exists()


def test_failing_run_returns_nonzero(tmp_path):
    wild = {"wild": {"provider": "mock", "temperature": 0.0, "cache": False,
                     "options": {"default_response": {"change_pct": 9000}}}}
    cfg = spec_file(tmp_path, llms=wild)
    assert main(["run", str(cfg), "--out", str(tmp_path / "out")]) == 1


def test_run_accepts_a_short_experiment_name(tmp_path, capsys):
    assert main(["run", "rbc/mock", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("[ok]") == 2 * 2 * 2 * 2
    assert (tmp_path / "rbc_mock" / "summary.parquet").exists()


def test_run_with_an_unknown_name_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "rbc/does_not_exist", "--out", str(tmp_path)])


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nope"])
