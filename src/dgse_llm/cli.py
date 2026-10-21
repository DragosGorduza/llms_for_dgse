"""Command line: dgse-llm run <config> | report <dir> | evaluate <run> | list."""

from __future__ import annotations

import argparse
from pathlib import Path

from dgse_llm.evaluation import AXES, evaluate_experiment
from dgse_llm.experiments import load_experiment, load_prompt_table, run_experiment
from dgse_llm.llms import LLM_REGISTRY
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import experiments_dir, runs_dir
from dgse_llm.reporting import write_experiment_report, write_run_report
from dgse_llm.shocks import MAPPING_REGISTRY, PROFILE_REGISTRY
from dgse_llm.solvers import SOLVER_REGISTRY


def _find_experiment(arg: str) -> Path:
    """A path, or a name under experiments/ such as `krusell_smith/generic`."""
    path = Path(arg)
    if path.exists():
        return path
    candidate = experiments_dir() / (arg if arg.endswith(".yaml") else f"{arg}.yaml")
    if candidate.exists():
        return candidate
    raise SystemExit(f"No experiment file '{arg}' (also looked for {candidate})")


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = load_experiment(_find_experiment(args.config))
    kept: set[str] = set()

    def on_skip(m) -> None:
        kept.add(m.run_id)
        print(f"[skip] {m.run_id}  already done, setup unchanged")

    manifests = run_experiment(
        cfg,
        out_root=args.out,
        on_cell=lambda m: print(f"[{m.status}] {m.run_id}" + (f"  {m.error}" if m.error else "")),
        on_skip=on_skip,
        force=args.force,
    )
    root = Path(args.out or cfg.out_dir or runs_dir()) / cfg.id
    failed = [m for m in manifests if m.status != "ok"]
    if failed:
        print(f"{len(failed)} run(s) failed; error, prompt and returned text are logged under {root / 'errors'}")
    if args.report:
        for m in manifests:
            if m.run_id not in kept or not (root / m.run_id / "report.md").exists():
                write_run_report(root / m.run_id)
        write_experiment_report(root)
        print(f"Reports written under {root}")
    if args.evaluate:
        print(f"Evaluation written to {evaluate_experiment(root)}")
    return 0 if not failed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    path = Path(args.directory)
    if (path / "index.parquet").exists():
        print(write_experiment_report(path))
    else:
        print(write_run_report(path))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    print(evaluate_experiment(args.run, axes=args.axes, out_dir=args.out))
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
        prompts = load_prompt_table(name)
        print(f"    prompts: {', '.join(f'{k} -> {v.variable}' for k, v in prompts.items())}")
        for s in model.classic_shocks:
            timing = {"one_time": "one period", "multi_period": f"{s.duration} periods", "geometric_decay": f"decay {s.rho}"}
            print(f"    classic shock {s.name}: {s.variable} {s.size:+g}, {timing[s.kind]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dgse-llm")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment file, e.g. experiments/nk3/mock.yaml or nk3/mock")
    run.add_argument("config")
    run.add_argument("--out", default=None, help="output root (default: runs/)")
    run.add_argument("--report", action="store_true", help="also write markdown reports")
    run.add_argument("--force", action="store_true", help="run every cell again, even if already done")
    run.add_argument("--evaluate", action="store_true", help="also write like-with-like comparisons to evaluations/")
    run.set_defaults(func=_cmd_run)

    report = sub.add_parser("report", help="write report.md for a run dir or experiment dir")
    report.add_argument("directory")
    report.set_defaults(func=_cmd_report)

    evaluate = sub.add_parser("evaluate", help="compare specifications of a run, e.g. ks_mock or runs/ks_mock")
    evaluate.add_argument("run")
    evaluate.add_argument("--axes", nargs="+", choices=AXES, default=None,
                          help="axes to compare (default: model_info llm persona scenario)")
    evaluate.add_argument("--out", default=None, help="output folder (default: <run>/evaluations)")
    evaluate.set_defaults(func=_cmd_evaluate)

    ls = sub.add_parser("list", help="list registered models, solvers, providers, profiles, mappings")
    ls.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
