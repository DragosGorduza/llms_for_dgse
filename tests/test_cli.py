"""Command line interface.

Expect: `list` prints the registries, `run --report` executes a config and writes reports, `report` works on a
run directory and on an experiment directory, and a failing run gives a non-zero exit code.
"""

import pytest

from dgse_llm.cli import main
from dgse_llm.paths import configs_dir
from dgse_llm.schemas import dump_yaml


def test_list(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for word in ("models:", "krusell_smith", "numpy_newton", "mock", "geometric_decay", "linear", "shockable inputs"):
        assert word in out


def test_run_and_report(tmp_path, capsys):
    cfg = configs_dir() / "experiments" / "nk_wedge_mock.yaml"
    assert main(["run", str(cfg), "--out", str(tmp_path), "--report"]) == 0
    out = capsys.readouterr().out
    assert out.count("[ok]") == 2
    exp = tmp_path / "nk_wedge_mock"
    assert (exp / "experiment_report.md").exists()
    run_dirs = [d for d in exp.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    assert len(run_dirs) == 2 and all((d / "report.md").exists() for d in run_dirs)


def test_report_command_picks_report_type(tmp_path):
    cfg = configs_dir() / "experiments" / "nk_wedge_mock.yaml"
    main(["run", str(cfg), "--out", str(tmp_path)])
    exp = tmp_path / "nk_wedge_mock"
    assert main(["report", str(exp)]) == 0 and (exp / "experiment_report.md").exists()
    run_dir = next(d for d in exp.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    assert main(["report", str(run_dir)]) == 0 and (run_dir / "report.md").exists()


def test_failing_run_returns_nonzero(tmp_path):
    cfg = tmp_path / "bad.yaml"
    dump_yaml(
        {
            "id": "bad", "T": 10, "models": ["nk3"],
            "scenarios": [{"id": "s", "shocks": [{"id": "b", "variable": "x", "model": "nk3"}]}],
        },
        cfg,
    )
    assert main(["run", str(cfg), "--out", str(tmp_path / "out")]) == 1


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["nope"])
