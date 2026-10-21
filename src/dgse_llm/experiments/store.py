"""Read and write run directories (see runs/README.md for the layout)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dgse_llm.schemas import LLMCall, RunManifest, ShockPath, dump_yaml, load_yaml

SHOCK_COLUMNS = ["shock_id", "variable", "mode", "t", "value", "active"]
TRAJ_COLUMNS = ["scenario", "variable", "t", "value"]
RUN_FILES = (
    "config.resolved.yaml", "manifest.json", "llm_calls.jsonl", "shock_paths.parquet", "trajectories.parquet",
    "metrics.json", "metrics_arms.json", "report.md",
)


class RunStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        """Delete the files of an earlier attempt so a failed re-run cannot leave stale results next to its manifest."""
        for name in RUN_FILES:
            (self.dir / name).unlink(missing_ok=True)

    def write_config(self, resolved: dict) -> None:
        dump_yaml(json.loads(json.dumps(resolved, default=str)), self.dir / "config.resolved.yaml")

    def write_manifest(self, manifest: RunManifest) -> None:
        (self.dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def write_llm_calls(self, calls: list[LLMCall]) -> None:
        lines = [c.model_dump_json() for c in calls]
        (self.dir / "llm_calls.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_shock_paths(self, paths: list[ShockPath]) -> None:
        rows = [
            {"shock_id": p.shock_id, "variable": p.variable, "mode": p.mode, "t": t, "value": v, "active": a}
            for p in paths
            for t, (v, a) in enumerate(zip(p.values, p.active))
        ]
        pd.DataFrame(rows, columns=SHOCK_COLUMNS).to_parquet(self.dir / "shock_paths.parquet", index=False)

    def write_trajectories(self, scenarios: dict[str, dict[str, np.ndarray]], variables: list[str]) -> None:
        rows = [
            {"scenario": s, "variable": v, "t": t, "value": float(x)}
            for s, paths in scenarios.items()
            for v in variables
            for t, x in enumerate(paths[v])
        ]
        pd.DataFrame(rows, columns=TRAJ_COLUMNS).to_parquet(self.dir / "trajectories.parquet", index=False)

    def write_metrics(self, metrics: dict) -> None:
        (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def write_arm_metrics(self, arm_metrics: dict) -> None:
        (self.dir / "metrics_arms.json").write_text(json.dumps(arm_metrics, indent=2), encoding="utf-8")


@dataclass
class RunData:
    run_dir: Path
    config: dict
    manifest: RunManifest
    llm_calls: list[LLMCall]
    shock_paths: pd.DataFrame
    trajectories: pd.DataFrame
    metrics: dict
    # {arm: {variable: metrics}} with arms classic, llm, classic_llm, interaction; empty unless the run had both kinds.
    arm_metrics: dict


def load_run(run_dir: str | Path) -> RunData:
    d = Path(run_dir)
    manifest = RunManifest.model_validate_json((d / "manifest.json").read_text(encoding="utf-8"))
    calls_file = d / "llm_calls.jsonl"
    calls = (
        [LLMCall.model_validate_json(line) for line in calls_file.read_text(encoding="utf-8").splitlines() if line]
        if calls_file.exists()
        else []
    )
    shocks_file, traj_file, metrics_file = d / "shock_paths.parquet", d / "trajectories.parquet", d / "metrics.json"
    arms_file = d / "metrics_arms.json"
    return RunData(
        run_dir=d,
        config=load_yaml(d / "config.resolved.yaml"),
        manifest=manifest,
        llm_calls=calls,
        shock_paths=pd.read_parquet(shocks_file) if shocks_file.exists() else pd.DataFrame(columns=SHOCK_COLUMNS),
        trajectories=pd.read_parquet(traj_file) if traj_file.exists() else pd.DataFrame(columns=TRAJ_COLUMNS),
        metrics=json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {},
        arm_metrics=json.loads(arms_file.read_text(encoding="utf-8")) if arms_file.exists() else {},
    )


def load_manifest(run_dir: str | Path) -> RunManifest | None:
    """The manifest of a run directory, or None if it is missing or unreadable."""
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def load_calls(run_dir: str | Path) -> list[LLMCall]:
    path = Path(run_dir) / "llm_calls.jsonl"
    if not path.exists():
        return []
    return [LLMCall.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "Error"


def write_error_log(exp_dir: str | Path, manifest: RunManifest, calls: list[LLMCall], kind: str, message: str) -> Path:
    """Log a failed run as errors/<ErrorName>/<run_id>.json: the error, the run's axes and every LLM call with the
    prompt that was sent and the response text that came back."""
    path = Path(exp_dir) / "errors" / _safe(kind) / f"{manifest.run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": manifest.run_id,
        "experiment_id": manifest.experiment_id,
        "error_type": kind,
        "error": message,
        "model": manifest.model,
        "llm": manifest.llm,
        "axes": manifest.meta,
        "finished": manifest.finished,
        "llm_calls": [
            {
                "prompt": c.prompt, "response_text": c.response_text, "parsed": c.parsed, "error": c.error,
                "provider": c.provider, "model_id": c.model_id, "temperature": c.temperature,
                "cache_hit": c.cache_hit, "prompt_hash": c.prompt_hash,
            }
            for c in calls
        ],
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def clear_error_logs(exp_dir: str | Path, run_id: str) -> None:
    """Remove the error log of a run that is about to be tried again, in whichever error folder it is."""
    for path in (Path(exp_dir) / "errors").glob(f"*/{run_id}.json"):
        path.unlink()


def write_index(exp_dir: str | Path, manifests: list[RunManifest]) -> Path:
    rows = []
    for m in manifests:
        row = m.model_dump(exclude={"diagnostics", "modified", "meta"})
        row.update(m.meta)
        row["modified"] = ", ".join(f"{x['variable']}({x['mode']})" for x in m.modified)
        shocked = m.diagnostics.get("shocked")
        row["converged"] = shocked.converged if shocked else None
        rows.append(row)
    path = Path(exp_dir) / "index.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
