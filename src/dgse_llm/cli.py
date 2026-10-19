"""Command line: dgse-llm run <config> | report <dir> | list."""

from __future__ import annotations

import argparse
from pathlib import Path

from dgse_llm.experiments import load_experiment, run_experiment
from dgse_llm.llms import LLM_REGISTRY
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import runs_dir
from dgse_llm.reporting import write_experiment_report, write_run_report
from dgse_llm.shocks import MAPPING_REGISTRY, PROFILE_REGISTRY
from dgse_llm.solvers import SOLVER_REGISTRY


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_experiment(args.config)
    manifests = run_experiment(
        cfg,
        out_root=args.out,
        on_cell=lambda m: print(f"[{m.status}] {m.run_id}" + (f"  {m.error}" if m.error else "")),
    )
    root = Path(args.out or cfg.out_dir or runs_dir()) / cfg.id
    if args.report:
        for m in manifests:
            write_run_report(root / m.run_id)
        write_experiment_report(root)
        print(f"Reports written under {root}")
    return 0 if all(m.status == "ok" for m in manifests) else 1


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.directory)
    if (path / "index.parquet").exists():
        print(write_experiment_report(path))
    else:
        print(write_run_report(path))
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    for label, registry in (
        ("models", MODEL_REGISTRY),
        ("solvers", SOLVER_REGISTRY),
        ("llm providers", LLM_REGISTRY),
        ("time profiles", PROFILE_REGISTRY),
        ("mappings", MAPPING_REGISTRY),
    ):
        print(f"{label}: {', '.join(registry.names())}")
    for name in MODEL_REGISTRY.names():
        model = MODEL_REGISTRY.get(name)()
        print(f"  {name} shockable inputs: {', '.join(model.shockable)}")
        for s in model.classic_shocks:
            timing = {"one_time": "one period", "multi_period": f"{s.duration} periods", "geometric_decay": f"decay {s.rho}"}
            print(f"    classic shock {s.name}: {s.variable} {s.size:+g}, {timing[s.kind]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dgse-llm")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config")
    run.add_argument("--out", default=None, help="output root (default: runs/)")
    run.add_argument("--report", action="store_true", help="also write markdown reports")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="write report.md for a run dir or experiment dir")
    report.add_argument("directory")
    report.set_defaults(func=_cmd_report)

    ls = sub.add_parser("list", help="list registered models, solvers, providers, profiles, mappings")
    ls.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
