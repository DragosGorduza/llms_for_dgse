"""Distribution of LLM answers and model outcomes across repeats, and the effect of showing the model to the LLM.

Reads only files in an experiment directory (index.parquet and each run's metrics.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AXES = ["prompt", "persona", "scenario"]
GROUP = ["model", "llm", *AXES, "model_info"]


def run_table(exp_dir: str | Path) -> pd.DataFrame:
    """One row per ok run: the index columns, the raw `answer` and `peak_<variable>` (peak impulse response)."""
    exp = Path(exp_dir)
    index = pd.read_parquet(exp / "index.parquet")
    index = index[index["status"] == "ok"].copy()
    peaks = []
    for run_id in index["run_id"]:
        metrics = json.loads((exp / run_id / "metrics.json").read_text(encoding="utf-8"))
        peaks.append({f"peak_{v}": vals["peak"] for v, vals in metrics.items()})
    return pd.concat([index.reset_index(drop=True), pd.DataFrame(peaks)], axis=1)


def build_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (model, llm, prompt, persona, scenario, model_info): n draws, answer distribution, mean peaks.

    answer_* columns describe the raw LLM answers (relative % change); peak_*_mean / peak_*_std the model outcomes.
    std is 0 for a single draw (temperature 0).
    """
    if runs.empty or "answer" not in runs or not set(GROUP) <= set(runs.columns):
        return pd.DataFrame()
    peak_cols = [c for c in runs.columns if c.startswith("peak_")]
    grouped = runs.groupby(GROUP, dropna=False)
    out = grouped["answer"].agg(
        n="count", answer_mean="mean", answer_std="std", answer_min="min", answer_median="median", answer_max="max"
    )
    for col in peak_cols:
        out[f"{col}_mean"] = grouped[col].mean()
        out[f"{col}_std"] = grouped[col].std()
    out["answer_std"] = out["answer_std"].fillna(0.0)
    for col in [c for c in out.columns if c.endswith("_std")]:
        out[col] = out[col].fillna(0.0)
    return out.reset_index()


def info_effect(summary: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison with and without the model info in the prompt, per prompt/persona/scenario/LLM.

    Columns: answer_without, answer_with (mean answers) and diff = with - without. Empty unless both were run.
    """
    if summary.empty:
        return pd.DataFrame()
    keys = ["model", "llm", *AXES]
    wide = summary.pivot_table(index=keys, columns="model_info", values="answer_mean", aggfunc="mean")
    if not {False, True} <= set(wide.columns):
        return pd.DataFrame()
    out = wide.rename(columns={False: "answer_without", True: "answer_with"})[["answer_without", "answer_with"]]
    out["diff"] = out["answer_with"] - out["answer_without"]
    return out.reset_index()


def write_summary(exp_dir: str | Path) -> pd.DataFrame:
    """Write summary.parquet (and info_effect.parquet when both settings were run) next to the index; returns the summary."""
    exp = Path(exp_dir)
    summary = build_summary(run_table(exp))
    if not summary.empty:
        summary.to_parquet(exp / "summary.parquet", index=False)
        effect = info_effect(summary)
        if not effect.empty:
            effect.to_parquet(exp / "info_effect.parquet", index=False)
    return summary
