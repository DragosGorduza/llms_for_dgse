"""Like-with-like comparisons of the runs of one experiment, one axis at a time.

An axis is one dimension of the experiment grid: llm, prompt, persona, scenario or model_info. Comparing along an axis
holds every other axis fixed, so two runs are only paired when they share model, llm, prompt, persona, scenario and
model info except for the axis itself (a "cell group"). Repeats (draws) of a cell are averaged first. A pair of levels
(a, b) is always in a fixed order (model_info: without -> with; otherwise alphabetical) and differences are b - a.

Reads only files in an experiment directory: index.parquet and, per ok run, metrics.json and trajectories.parquet.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

AXES = ("llm", "prompt", "persona", "scenario", "model_info")
# Comparing prompts compares different shocked inputs, so it is not part of the default like-with-like set.
DEFAULT_AXES = ("model_info", "llm", "persona", "scenario")
CELL = ("model", *AXES)
METRICS = ("peak", "peak_t", "cumulative", "half_life")
ANSWER = "llm_answer"
ANSWER_METRIC = "change_pct"
LEVEL_ORDER = {"model_info": ["without", "with"]}

PAIR_COLUMNS = ["a", "b", "a_mean", "b_mean", "a_n", "b_n", "diff", "rel_diff", "z", "sign_flip"]
TRAJ_PAIR_COLUMNS = ["a", "b", "rmse", "rel_rmse", "max_abs_diff", "max_abs_diff_t", "corr", "cum_diff"]


def load_runs(exp_dir: str | Path) -> pd.DataFrame:
    """One row per ok run: run_id, model, the AXES, repeat and answer (float, NaN if missing).

    Every axis column is str; model_info becomes 'without' / 'with'.
    """
    exp = Path(exp_dir)
    index = pd.read_parquet(exp / "index.parquet")
    missing = [c for c in (*CELL, "repeat", "answer") if c not in index.columns]
    if missing:
        raise ValueError(f"{exp / 'index.parquet'} has no column(s) {missing}; evaluate the runs of an experiment file")
    runs = index.loc[index["status"] == "ok", ["run_id", *CELL, "repeat", "answer"]].reset_index(drop=True)
    runs["model_info"] = runs["model_info"].map({True: "with", False: "without"})
    for col in CELL:
        runs[col] = runs[col].fillna("none").astype(str)
    runs["answer"] = pd.to_numeric(runs["answer"], errors="coerce")
    return runs


def metric_table(exp_dir: str | Path, runs: pd.DataFrame) -> pd.DataFrame:
    """Long table, one row per run x variable x metric: run columns plus variable, metric, value (float).

    The raw LLM answer is the row variable=ANSWER, metric=ANSWER_METRIC; impulse-response metrics come from metrics.json.
    """
    exp = Path(exp_dir)
    rows = []
    for run_id, answer in zip(runs["run_id"], runs["answer"]):
        rows.append({"run_id": run_id, "variable": ANSWER, "metric": ANSWER_METRIC, "value": answer})
        metrics = json.loads((exp / run_id / "metrics.json").read_text(encoding="utf-8"))
        for variable, values in metrics.items():
            rows.extend(
                {"run_id": run_id, "variable": variable, "metric": m, "value": values.get(m)} for m in METRICS
            )
    long = pd.DataFrame(rows, columns=["run_id", "variable", "metric", "value"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return runs.drop(columns="answer").merge(long, on="run_id")


def irf_table(exp_dir: str | Path, runs: pd.DataFrame) -> pd.DataFrame:
    """Long table of impulse responses (shocked minus baseline): run columns plus variable, t, value."""
    exp = Path(exp_dir)
    frames = []
    for run_id in runs["run_id"]:
        traj = pd.read_parquet(exp / run_id / "trajectories.parquet")
        frames.append(traj.loc[traj["scenario"] == "irf", ["variable", "t", "value"]].assign(run_id=run_id))
    columns = [*runs.columns.drop("answer"), "variable", "t", "value"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return runs.drop(columns="answer").merge(pd.concat(frames, ignore_index=True), on="run_id")[columns]


def cell_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Mean, std (0 for one draw) and n over the repeats of each cell, per variable and metric."""
    out = long.groupby([*CELL, "variable", "metric"], sort=True)["value"].agg(mean="mean", std="std", n="count")
    out["std"] = out["std"].fillna(0.0)
    return out.reset_index()


def irf_cells(irf: pd.DataFrame) -> pd.DataFrame:
    """Mean impulse response of each cell over its repeats, with the lowest and highest draw (`low`, `high`)."""
    out = irf.groupby([*CELL, "variable", "t"], sort=True)["value"].agg(mean="mean", low="min", high="max", n="count")
    return out.reset_index()


def ordered_levels(axis: str, values: Iterable) -> list[str]:
    """Distinct levels of an axis in comparison order: LEVEL_ORDER first, the rest alphabetically."""
    present = set(values)
    order = [x for x in LEVEL_ORDER.get(axis, []) if x in present]
    return order + sorted(present - set(order))


def _held_fixed(axis: str) -> list[str]:
    return [c for c in CELL if c != axis]


def pair_table(cells: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Every pair of levels of `axis` inside every cell group, per variable and metric.

    Columns: the held-fixed axes, variable, metric, a, b, a_mean, b_mean, a_n, b_n, diff (b - a), rel_diff
    (diff / |a_mean|), z (diff over the standard error from the repeats; NaN with one draw each) and sign_flip.
    """
    keys = [*_held_fixed(axis), "variable", "metric"]
    rows = []
    for group, frame in cells.groupby(keys, sort=True):
        by_level = {rec[axis]: rec for rec in frame.to_dict("records")}
        for a, b in itertools.combinations(ordered_levels(axis, by_level), 2):
            ra, rb = by_level[a], by_level[b]
            diff = rb["mean"] - ra["mean"]
            se = math.sqrt(ra["std"] ** 2 / max(ra["n"], 1) + rb["std"] ** 2 / max(rb["n"], 1))
            rows.append({
                **dict(zip(keys, group)), "a": a, "b": b, "a_mean": ra["mean"], "b_mean": rb["mean"],
                "a_n": ra["n"], "b_n": rb["n"], "diff": diff,
                "rel_diff": diff / abs(ra["mean"]) if ra["mean"] != 0 else np.nan,
                "z": diff / se if se > 0 else np.nan,
                "sign_flip": bool(ra["mean"] * rb["mean"] < 0),
            })
    return pd.DataFrame(rows, columns=[*keys, *PAIR_COLUMNS])


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (a, b, variable, metric) over all cell groups: n_cells, mean_a, mean_b, mean/median/min/max diff,
    mean_abs_diff, share_b_higher (share of groups with b > a) and sign_flips (groups where the sign differs)."""
    if pairs.empty:
        return pd.DataFrame()
    diff = pairs["diff"]
    work = pairs.assign(abs_diff=diff.abs(), b_higher=np.where(diff.isna(), np.nan, (diff > 0).astype(float)))
    out = work.groupby(["a", "b", "variable", "metric"], sort=False).agg(
        n_cells=("diff", "count"), mean_a=("a_mean", "mean"), mean_b=("b_mean", "mean"),
        mean_diff=("diff", "mean"), median_diff=("diff", "median"), min_diff=("diff", "min"),
        max_diff=("diff", "max"), mean_abs_diff=("abs_diff", "mean"), share_b_higher=("b_higher", "mean"),
        sign_flips=("sign_flip", "sum"),
    )
    return out.reset_index()


def trajectory_pairs(irf: pd.DataFrame, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Distance between the mean impulse responses of every pair of levels inside every cell group.

    Returns (pairs, diffs). pairs: held-fixed axes, variable, a, b, rmse, rel_rmse (rmse over the largest |response|
    of the two), max_abs_diff and its period, corr (NaN if a path is flat), cum_diff (sum of b - a). diffs: long table
    a, b, variable, t, diff of every paired path, for averaging over groups.
    """
    keys = [*_held_fixed(axis), "variable"]
    rows, diff_frames = [], []
    for group, frame in irf.groupby(keys, sort=True):
        paths = {lvl: f.sort_values("t") for lvl, f in frame.groupby(axis)}
        for a, b in itertools.combinations(ordered_levels(axis, paths), 2):
            xa, xb = paths[a]["mean"].to_numpy(float), paths[b]["mean"].to_numpy(float)
            if len(xa) != len(xb) or len(xa) == 0:
                continue
            d = xb - xa
            rmse = float(np.sqrt(np.mean(d ** 2)))
            scale = float(max(np.abs(xa).max(), np.abs(xb).max()))
            corr = float(np.corrcoef(xa, xb)[0, 1]) if xa.std() > 0 and xb.std() > 0 else np.nan
            rows.append({
                **dict(zip(keys, group)), "a": a, "b": b, "rmse": rmse,
                "rel_rmse": rmse / scale if scale > 0 else np.nan, "max_abs_diff": float(np.abs(d).max()),
                "max_abs_diff_t": int(paths[a]["t"].iloc[int(np.argmax(np.abs(d)))]), "corr": corr,
                "cum_diff": float(d.sum()),
            })
            diff_frames.append(pd.DataFrame({"a": a, "b": b, "variable": group[-1], "t": paths[a]["t"].to_numpy(),
                                             "diff": d}))
    pairs = pd.DataFrame(rows, columns=[*keys, *TRAJ_PAIR_COLUMNS])
    diffs = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame(
        columns=["a", "b", "variable", "t", "diff"])
    return pairs, diffs


def summarize_trajectory_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per (a, b, variable) over all cell groups: n_cells, mean rmse, rel_rmse, corr and cum_diff, max of max_abs_diff."""
    if pairs.empty:
        return pd.DataFrame()
    out = pairs.groupby(["a", "b", "variable"], sort=False).agg(
        n_cells=("rmse", "count"), mean_rmse=("rmse", "mean"), mean_rel_rmse=("rel_rmse", "mean"),
        mean_corr=("corr", "mean"), mean_cum_diff=("cum_diff", "mean"), max_abs_diff=("max_abs_diff", "max"),
    )
    return out.reset_index()


def mean_difference(diffs: pd.DataFrame) -> pd.DataFrame:
    """Per (a, b, variable, t): mean of b - a over the cell groups, with the lowest and highest group (`low`, `high`)."""
    if diffs.empty:
        return pd.DataFrame(columns=["a", "b", "variable", "t", "mean", "low", "high", "n"])
    out = diffs.groupby(["a", "b", "variable", "t"], sort=False)["diff"].agg(
        mean="mean", low="min", high="max", n="count")
    return out.reset_index()


def axis_importance(cells: pd.DataFrame, axes: Iterable[str]) -> pd.DataFrame:
    """How much each axis moves results, holding the others fixed.

    For every cell group with at least two levels of the axis: range = max - min of the cell means across levels and
    rel_range = range / mean |cell mean|. Per axis, variable and metric: n_cells, mean_range, median_range,
    mean_rel_range.
    """
    frames = []
    for axis in axes:
        keys = [*_held_fixed(axis), "variable", "metric"]
        g = cells.assign(abs_mean=cells["mean"].abs()).groupby(keys, sort=True).agg(
            n_levels=("mean", "count"), hi=("mean", "max"), lo=("mean", "min"), scale=("abs_mean", "mean"))
        g = g[g["n_levels"] >= 2]
        if g.empty:
            continue
        g = g.assign(range=g["hi"] - g["lo"])
        g["rel_range"] = g["range"] / g["scale"].where(g["scale"] > 0)
        out = g.reset_index().groupby(["variable", "metric"], sort=False).agg(
            n_cells=("range", "count"), mean_range=("range", "mean"), median_range=("range", "median"),
            mean_rel_range=("rel_range", "mean"))
        frames.append(out.reset_index().assign(axis=axis))
    if not frames:
        return pd.DataFrame()
    cols = ["axis", "variable", "metric", "n_cells", "mean_range", "median_range", "mean_rel_range"]
    return pd.concat(frames, ignore_index=True)[cols]
