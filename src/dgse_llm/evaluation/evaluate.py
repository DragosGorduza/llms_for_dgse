"""Evaluate an experiment run: like-with-like comparison tables, figures and evaluation.md under <run>/evaluations/.

Usage: `evaluate_experiment("ks_mock")` (a name under runs/) or `evaluate_experiment("runs/ks_mock")`; from the
command line `dgse-llm evaluate ks_mock`, or `dgse-llm run <experiment> --evaluate` right after a run.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from dgse_llm.evaluation.compare import (
    ANSWER,
    ANSWER_METRIC,
    AXES,
    CELL,
    DEFAULT_AXES,
    axis_importance,
    cell_stats,
    irf_cells,
    irf_table,
    load_runs,
    mean_difference,
    metric_table,
    ordered_levels,
    pair_table,
    summarize_pairs,
    summarize_trajectory_pairs,
    trajectory_pairs,
)
from dgse_llm.paths import runs_dir
from dgse_llm.reporting.report import (
    _add_shock_markers,
    _grid,
    _save,
    _shock_markers,
    _style_axis,
    md_table,
)
from dgse_llm.reporting.style import load_style, model_color
from dgse_llm.schemas import load_yaml

AXIS_TITLES = {
    "model_info": "Model information in the prompt",
    "llm": "LLM",
    "persona": "Persona",
    "scenario": "Scenario (narrative and shocks)",
    "prompt": "Prompt (shocked input)",
}
REPORT_METRICS = (ANSWER_METRIC, "peak", "cumulative")
SCATTER_METRICS = ("peak", "cumulative")


def experiment_dir(target: str | Path) -> Path:
    """An experiment run directory (holding index.parquet): a path, or a run name under runs/."""
    for candidate in (Path(target), runs_dir() / str(target)):
        if (candidate / "index.parquet").exists():
            return candidate
    raise FileNotFoundError(f"No experiment run '{target}': no index.parquet in {target} or {runs_dir() / str(target)}")


def _slug(parts) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", str(p)) for p in parts)[:150] or "all"


def _look(i: int, style: dict) -> tuple[str, float]:
    """Line style and shade of the i-th level: line styles first, then fainter shades, so up to 16 are distinct."""
    styles, alphas = style["run_linestyles"], style["run_alphas"]
    return styles[i % len(styles)], alphas[(i // len(styles)) % len(alphas)]


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.unlink(missing_ok=True)
    else:
        df.to_parquet(path, index=False)


def _markers(exp: Path, runs: pd.DataFrame, keys: list[str], group: tuple) -> list[tuple[int, str]]:
    """Shock onsets of the first run of a cell group (all runs in the group share the scenario)."""
    mask = np.logical_and.reduce([runs[k] == v for k, v in zip(keys, group)])
    run = exp / runs.loc[mask, "run_id"].iloc[0]
    return _shock_markers(pd.read_parquet(run / "shock_paths.parquet"), load_yaml(run / "config.resolved.yaml"))


def _plot_cell_group(frame, axis, levels, variables, title, path, model, markers, style) -> None:
    """Mean impulse response of each level (band: lowest to highest draw when there are repeats)."""
    color = model_color(model, style)
    fig, axes = _grid(variables, title)
    for ax, variable in zip(axes, variables):
        for i, level in enumerate(levels):
            d = frame[(frame[axis] == level) & (frame["variable"] == variable)].sort_values("t")
            if d.empty:
                continue
            linestyle, alpha = _look(i, style)
            ax.plot(d["t"], d["mean"], color=color, linestyle=linestyle, alpha=alpha, linewidth=1.8,
                    label=f"{axis} = {level}")
            if (d["high"] > d["low"]).any():
                ax.fill_between(d["t"], d["low"], d["high"], color=color, alpha=style["band_alpha"] * alpha)
        _add_shock_markers(ax, markers)
        ax.axhline(0, color=style["neutral_color"], linestyle="--", linewidth=1, label="base-model")
        _style_axis(ax, variable)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
    _save(fig, path)


def _plot_mean_difference(frame, variables, title, path, model, style) -> None:
    """Mean over cell groups of the paired difference b - a of the impulse responses (band: lowest to highest group)."""
    color = model_color(model, style)
    fig, axes = _grid(variables, title)
    for ax, variable in zip(axes, variables):
        d = frame[frame["variable"] == variable].sort_values("t")
        ax.plot(d["t"], d["mean"], color=color, linewidth=1.8, label="mean difference")
        ax.fill_between(d["t"], d["low"], d["high"], color=color, alpha=style["band_alpha"],
                        label="range over cells")
        ax.axhline(0, color=style["neutral_color"], linestyle="--", linewidth=1, label="no difference")
        _style_axis(ax, variable)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
    _save(fig, path)


def _plot_scatter(pairs, a, b, metric, variables, path, model, style) -> None:
    """One point per cell group: value under level a (x) against level b (y); the LLM answer is the first panel."""
    color, fonts = model_color(model, style), style["fonts"]
    panels = [(ANSWER, ANSWER_METRIC), *((v, metric) for v in variables)]
    fig, axes = _grid(panels, f"{metric}: {b} (y) against {a} (x), one point per like-for-like cell")
    for ax, (variable, m) in zip(axes, panels):
        d = pairs[(pairs["a"] == a) & (pairs["b"] == b) & (pairs["variable"] == variable) & (pairs["metric"] == m)]
        d = d.dropna(subset=["a_mean", "b_mean"])
        if d.empty:
            ax.set_visible(False)
            continue
        ax.scatter(d["a_mean"], d["b_mean"], color=color)
        lo = float(min(d["a_mean"].min(), d["b_mean"].min()))
        hi = float(max(d["a_mean"].max(), d["b_mean"].max()))
        ax.plot([lo, hi], [lo, hi], color=style["neutral_color"], linestyle="--", linewidth=1, label="b = a")
        _style_axis(ax, "LLM answer (% change)" if variable == ANSWER else f"{variable} ({m})")
        ax.set_xlabel(str(a), fontsize=fonts["axis_label"])
        ax.set_ylabel(str(b), fontsize=fonts["axis_label"])
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    _save(fig, path)


def _evaluate_axis(axis, exp, out, runs, cells, irf, variables, model, style) -> list[str]:
    """Tables and figures for one axis under out/<axis>/; returns the markdown section."""
    axis_dir = out / axis
    fig_dir = axis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob("*.png"):
        old.unlink()
    pairs = pair_table(cells, axis)
    summary = summarize_pairs(pairs)
    tpairs, diffs = trajectory_pairs(irf, axis)
    tsummary = summarize_trajectory_pairs(tpairs)
    mean_diff = mean_difference(diffs)
    for df, name in ((pairs, "pairs"), (summary, "pairs_summary"), (tpairs, "trajectory_pairs"),
                     (tsummary, "trajectory_pairs_summary"), (mean_diff, "mean_difference")):
        _write_parquet(df, axis_dir / f"{name}.parquet")

    held = [c for c in CELL if c not in (axis, "model")]
    levels = ordered_levels(axis, runs[axis])
    lines = [f"## {AXIS_TITLES.get(axis, axis)} (`{axis}`)\n",
             f"Levels: {', '.join(f'`{x}`' for x in levels)}. Each pair (a, b) is compared inside cells that share "
             f"{', '.join(f'`{c}`' for c in held)}; differences are b - a.\n"]
    if pairs.empty:
        return lines + ["No like-for-like pairs: no cell group has two levels of this axis.\n"]

    lines.append("### LLM answers and model outcomes\n")
    lines.append("`change_pct` is the raw LLM answer; `peak` and `cumulative` describe the impulse response of each "
                 "variable. Means over cells; `share_b_higher` is the share of cells where b > a; `sign_flips` counts "
                 "cells where a and b have opposite signs.\n")
    rows = summary[summary["metric"].isin(REPORT_METRICS)].to_dict("records")
    lines.append(md_table(rows, ["a", "b", "variable", "metric", "n_cells", "mean_a", "mean_b", "mean_diff",
                                 "mean_abs_diff", "share_b_higher", "sign_flips"]))
    if not tsummary.empty:
        lines.append("### Distance between impulse responses\n")
        lines.append("`rel_rmse` is the RMSE of b - a over the larger peak of the two paths; `mean_corr` the mean "
                     "correlation of the two paths; `mean_cum_diff` the mean summed difference.\n")
        lines.append(md_table(tsummary.to_dict("records"), ["a", "b", "variable", "n_cells", "mean_rmse",
                                                             "mean_rel_rmse", "mean_corr", "mean_cum_diff",
                                                             "max_abs_diff"]))
    answers = pairs[pairs["variable"] == ANSWER]
    if not answers.empty:
        lines.append("### LLM answer per cell\n")
        lines.append("`z` is the difference over its standard error from the repeats (empty with one draw).\n")
        lines.append(md_table(answers.to_dict("records"), [*held, "a", "b", "a_mean", "b_mean", "diff", "z"]))

    lines.append("### Figures\n")
    for a, b in dict.fromkeys(zip(summary["a"], summary["b"])):
        stem = _slug([a, b])
        d = mean_diff[(mean_diff["a"] == a) & (mean_diff["b"] == b)]
        if not d.empty:
            name = f"mean_difference__{stem}.png"
            _plot_mean_difference(d, variables, f"Impulse response {b} minus {a}, mean over like-for-like cells",
                                  fig_dir / name, model, style)
            lines.append(f"![{b} minus {a}]({axis}/figures/{name})\n")
        for metric in SCATTER_METRICS:
            name = f"scatter_{metric}__{stem}.png"
            _plot_scatter(pairs, a, b, metric, variables, fig_dir / name, model, style)
            lines.append(f"![{metric}: {b} against {a}]({axis}/figures/{name})\n")

    keys = [c for c in CELL if c != axis]
    links = []
    for group, frame in irf.groupby(keys, sort=True):
        present = ordered_levels(axis, frame[axis])
        if len(present) < 2:
            continue
        fixed = {k: v for k, v in zip(keys, group) if k != "model"}
        title = f"{AXIS_TITLES.get(axis, axis)}: " + ", ".join(f"{k}={v}" for k, v in fixed.items())
        markers = [] if axis == "scenario" else _markers(exp, runs, keys, group)
        name = f"cell__{_slug(fixed.values())}.png"
        _plot_cell_group(frame, axis, present, variables, title, fig_dir / name, group[0], markers, style)
        links.append(f"- [{', '.join(f'{k}={v}' for k, v in fixed.items())}]({axis}/figures/{name})")
    if links:
        lines.append("Impulse responses per cell (one figure per like-for-like cell, one line per level):\n")
        lines.append("\n".join(links) + "\n")
    return lines


def evaluate_experiment(target: str | Path, axes: list[str] | None = None, out_dir: str | Path | None = None) -> Path:
    """Compare the specifications of an experiment run along each axis and write <run>/evaluations/evaluation.md.

    `target` is a run name under runs/ (for example `ks_mock`) or a path to an experiment run directory. `axes`
    defaults to DEFAULT_AXES; axes with a single level are skipped. Returns the path of evaluation.md.
    """
    axes = list(axes or DEFAULT_AXES)
    unknown = [a for a in axes if a not in AXES]
    if unknown:
        raise ValueError(f"Unknown axes {unknown}; choose from {list(AXES)}")
    exp = experiment_dir(target)
    out = Path(out_dir) if out_dir else exp / "evaluations"
    out.mkdir(parents=True, exist_ok=True)
    style = load_style()
    runs = load_runs(exp)
    text = [f"# Evaluation of experiment `{exp.name}`\n"]
    if runs.empty:
        text.append("No successful runs to compare.\n")
        return _write_report(out, text)

    cells = cell_stats(metric_table(exp, runs))
    irf = irf_cells(irf_table(exp, runs))
    _write_parquet(cells, out / "cells.parquet")
    _write_parquet(irf, out / "irf_cells.parquet")
    variables = list(dict.fromkeys(irf["variable"]))
    model = runs["model"].iloc[0]
    compared = [a for a in axes if runs[a].nunique() > 1]
    skipped = [a for a in axes if a not in compared]

    text.append(f"{len(runs)} successful runs of model(s) {', '.join(sorted(set(runs['model'])))}. Every comparison "
                "is like with like: runs are paired only when all other axes are equal, and repeats of a cell are "
                "averaged first. Differences are b - a.\n")
    text.append("Compared axes: " + (", ".join(f"`{a}`" for a in compared) or "none") + ".")
    if skipped:
        text.append(" Skipped (one level only): " + ", ".join(f"`{a}`" for a in skipped) + ".")
    text.append("\n")

    importance = axis_importance(cells, compared)
    _write_parquet(importance, out / "axis_importance.parquet")
    if not importance.empty:
        text.append("## Which axis moves results most\n")
        text.append("Per cell with every other axis fixed: `range` = largest minus smallest mean across the axis's "
                    "levels; `rel_range` = range over the mean absolute value. Averaged over cells.\n")
        rows = importance[importance["metric"].isin(REPORT_METRICS)].sort_values(["metric", "variable", "axis"])
        text.append(md_table(rows.to_dict("records"), ["variable", "metric", "axis", "n_cells", "mean_range",
                                                       "median_range", "mean_rel_range"]))
    for axis in compared:
        text.extend(_evaluate_axis(axis, exp, out, runs, cells, irf, variables, model, style))

    text.append("## Files\n")
    text.append("`cells.parquet` (mean, std, n per cell, variable and metric), `irf_cells.parquet` (mean impulse "
                "response per cell), `axis_importance.parquet`, and per axis `pairs.parquet`, `pairs_summary.parquet`, "
                "`trajectory_pairs.parquet`, `trajectory_pairs_summary.parquet`, `mean_difference.parquet` and "
                "`figures/`.\n")
    return _write_report(out, text)


def _write_report(out: Path, text: list[str]) -> Path:
    path = out / "evaluation.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path
