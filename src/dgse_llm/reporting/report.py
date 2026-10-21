"""Markdown reports built only from files in a run or experiment directory."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from dgse_llm.experiments.store import RunData, load_run  # noqa: E402
from dgse_llm.experiments.summary import AXES, build_summary, info_effect, run_table  # noqa: E402
from dgse_llm.reporting.style import (  # noqa: E402
    arm_label,
    arm_style,
    load_style,
    model_color,
    run_alpha,
    run_label,
    run_linestyle,
    shock_kinds,
)

MAX_TEXT = 700


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c)).replace("|", "\\|") for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_TEXT else text[:MAX_TEXT] + " ... [truncated; full text in llm_calls.jsonl]"


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in _truncate(text).splitlines()) or ">"


def _grid(variables: list[str], title: str):
    style = load_style()
    size = style["figure"]
    n = max(1, len(variables))
    cols = min(int(size["max_columns"]), n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(size["panel_width"] * cols, size["panel_height"] * rows), squeeze=False
    )
    fig.suptitle(title, fontsize=style["fonts"]["suptitle"])
    return fig, axes.ravel()


def _save(fig, path: Path) -> None:
    """Tidy the layout, save with the configured dpi and close the figure."""
    fig.tight_layout()
    fig.savefig(path, dpi=load_style()["figure"]["dpi"])
    plt.close(fig)


def _style_axis(ax, title: str, legend: bool = True) -> None:
    """Panel title, axis label, tick labels and legend, with the font sizes from plotting.yaml."""
    fonts = load_style()["fonts"]
    ax.set_title(title, fontsize=fonts["title"])
    ax.set_xlabel("t", fontsize=fonts["axis_label"])
    ax.tick_params(labelsize=fonts["tick"])
    if legend and ax.get_lines():
        ax.legend(fontsize=fonts["legend"])


ARM_ORDER = ("baseline", "classic", "llm", "shocked")
# Which stored impulse-response scenario belongs to each (non-baseline) arm.
IRF_SCENARIO = {"classic": "irf_classic", "llm": "irf_llm", "shocked": "irf"}


def _plot_trajectories(
    traj: pd.DataFrame, path: Path, model: str, kinds: list[str],
    shocks: pd.DataFrame | None = None, config: dict | None = None,
) -> None:
    style = load_style()
    markers = _shock_markers(shocks, config or {}) if shocks is not None else []
    arms = [a for a in ARM_ORDER if a in set(traj["scenario"])]
    variables = list(dict.fromkeys(traj[traj["scenario"] == "shocked"]["variable"]))
    fig, axes = _grid(variables, "Model paths: base-model versus shocked economies")
    for ax, variable in zip(axes, variables):
        for arm in arms:
            values = traj[(traj["scenario"] == arm) & (traj["variable"] == variable)].sort_values("t")
            ax.plot(values["t"], values["value"], label=arm_label(arm, kinds), **arm_style(arm, model, style))
        _add_shock_markers(ax, markers)
        _style_axis(ax, variable)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
    _save(fig, path)


def _shock_markers(shocks: pd.DataFrame, config: dict) -> list[tuple[int, str]]:
    """Return shock onset periods and readable labels from the saved paths and resolved config."""
    if shocks.empty:
        return []

    specs = {spec.get("id"): spec for spec in config.get("scenario", {}).get("shocks", [])}
    markers = set()
    for (shock_id, variable, _mode), rows in shocks.groupby(
        ["shock_id", "variable", "mode"], sort=False
    ):
        spec = specs.get(shock_id, {})
        occurrences = (spec.get("llm") or {}).get("occurrences", [])
        if occurrences:
            active_periods = set(rows.loc[rows["active"], "t"].astype(int))
            for occurrence in occurrences:
                periods = occurrence.get("periods", [])
                if not periods:
                    continue
                period = min(periods)
                if period in active_periods:
                    name = occurrence.get("name") or shock_id
                    markers.add((int(period), f"Shock: {name} ({variable})"))
            continue

        profile = spec.get("profile") or {}
        if profile.get("kind") == "at_periods":
            active_periods = set(rows.loc[rows["active"], "t"].astype(int))
            starts = [period for period in profile.get("periods", []) if period in active_periods]
        else:
            ordered = rows.sort_values("t")
            active = ordered["active"].fillna(False).astype(bool)
            starts = ordered.loc[active & ~active.shift(1, fill_value=False), "t"].tolist()
        name = spec.get("classic") or shock_id
        markers.update((int(period), f"Shock: {name} ({variable})") for period in starts)
    return sorted(markers)


def _add_shock_markers(ax, markers: list[tuple[int, str]]) -> None:
    color = load_style()["event_color"]
    for period, label in markers:
        ax.axvline(period, color=color, linestyle="--", linewidth=1, alpha=0.7, label=label)


def _plot_irfs(
    traj: pd.DataFrame, path: Path, model: str, kinds: list[str],
    shocks: pd.DataFrame | None = None, config: dict | None = None,
) -> None:
    style = load_style()
    markers = _shock_markers(shocks, config or {}) if shocks is not None else []
    present = set(traj["scenario"])
    arms = [a for a in ("classic", "llm", "shocked") if IRF_SCENARIO[a] in present]
    variables = list(dict.fromkeys(traj[traj["scenario"] == "irf"]["variable"]))
    fig, axes = _grid(variables, "Impulse responses (each economy minus the base-model)")
    for ax, v in zip(axes, variables):
        for arm in arms:
            d = traj[(traj["scenario"] == IRF_SCENARIO[arm]) & (traj["variable"] == v)].sort_values("t")
            effect = arm_label(arm, kinds).removeprefix("base-model + ")
            ax.plot(d["t"], d["value"], label=effect, **arm_style(arm, model, style))
        _add_shock_markers(ax, markers)
        ax.axhline(0, color=style["neutral_color"], linestyle="--", linewidth=1, label="base-model")
        _style_axis(ax, v)
    for ax in axes[len(variables):]:
        ax.set_visible(False)
    _save(fig, path)


def _plot_shocks(shocks: pd.DataFrame, path: Path, color: str) -> None:
    keys = list(dict.fromkeys(zip(shocks["shock_id"], shocks["variable"], shocks["mode"])))
    fig, axes = _grid(keys, "Injected shock paths")
    for ax, (sid, var, mode) in zip(axes, keys):
        d = shocks[(shocks["shock_id"] == sid) & (shocks["variable"] == var)]
        ax.step(d["t"], d["value"], where="post", color=color)
        _style_axis(ax, f"{sid}: {var} ({mode})", legend=False)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    _save(fig, path)


def write_run_report(run_dir: str | Path) -> Path:
    run = load_run(run_dir)
    d, m = run.run_dir, run.manifest
    fig_dir = d / "figures"
    fig_dir.mkdir(exist_ok=True)
    cfg = run.config
    color = model_color(m.model)
    kinds = shock_kinds(m)
    out = [f"# Run `{m.run_id}`\n"]
    out.append(f"Experiment `{m.experiment_id}`. Status: **{m.status}**.\n")
    if m.error:
        out.append(f"Error: `{m.error}`\n")

    out.append("## Setup\n")
    out.append(
        md_table(
            [{
                "model": m.model, "solver": cfg["model"]["solver"], "llm": m.llm, "scenario": m.scenario,
                "intensity": m.intensity, "mode": m.mode, "repeat": m.repeat, "T": cfg["T"],
            }],
            ["model", "solver", "llm", "scenario", "intensity", "mode", "repeat", "T"],
        )
    )
    if cfg["model"]["params"]:
        out.append("Parameter overrides: " + ", ".join(f"`{k}={v}`" for k, v in cfg["model"]["params"].items()) + "\n")
    if m.meta:
        out.append(md_table([{**m.meta, "answer": m.answer}], [*m.meta, "answer"]))

    out.append("## Modified variables\n")
    out.append(md_table(m.modified, ["shock_id", "variable", "mode", "source"]))

    if len(run.shock_paths):
        _plot_shocks(run.shock_paths, fig_dir / "shock_paths.png", color)
        out.append("![shock paths](figures/shock_paths.png)\n")

    out.append(f"## LLM calls ({len(run.llm_calls)})\n")
    for call in run.llm_calls:
        out.append(
            f"### `{call.id}` shock `{call.shock_id}`, t={call.context.get('t')}, "
            f"{call.provider}/{call.model_id}, cache hit: {call.cache_hit}\n"
        )
        out.append("Prompt:\n\n" + _quote(call.prompt) + "\n")
        out.append("Answer:\n\n" + _quote(call.response_text) + "\n")
        out.append(f"Parsed: `{call.parsed}`" + (f". Error: `{call.error}`" if call.error else "") + "\n")

    if len(run.trajectories):
        _plot_trajectories(run.trajectories, fig_dir / "trajectories.png", m.model, kinds,
                   run.shock_paths, cfg)
        out.append("## Model paths\n\nEach shocked economy against the base-model (all shocks removed).\n\n"
                   "![model paths](figures/trajectories.png)\n")
        _plot_irfs(run.trajectories, fig_dir / "irf.png", m.model, kinds, run.shock_paths, cfg)
        out.append("## Impulse responses\n\n![irf](figures/irf.png)\n")
        rows = [{"variable": v, **vals} for v, vals in run.metrics.items()]
        out.append("## Metrics\n")
        out.append(md_table(rows, ["variable", "peak", "peak_t", "cumulative", "half_life"]))
        if run.arm_metrics:
            out.append("## Classic shock, LLM shock and both together\n\n"
                       "Four economies: base-model, + classic shock, + LLM shock, + both. `interaction` is the effect "
                       "of combining them beyond the sum of the two separate effects (0 in a linear model).\n")
            for metric in ("peak", "cumulative"):
                out.append(f"### {metric} response\n")
                arm_rows = [
                    {
                        "variable": v,
                        "classic shock": run.arm_metrics["classic"][v][metric],
                        "LLM shock": run.arm_metrics["llm"][v][metric],
                        "classic + LLM": run.arm_metrics["classic_llm"][v][metric],
                        "interaction": run.arm_metrics["interaction"][v][metric],
                    }
                    for v in run.metrics
                ]
                out.append(md_table(arm_rows, ["variable", "classic shock", "LLM shock", "classic + LLM", "interaction"]))

    out.append("## Solver diagnostics\n")
    out.append(
        md_table(
            [{"run": k, **v.model_dump()} for k, v in m.diagnostics.items()],
            ["run", "solver", "converged", "iterations", "residual_norm"],
        )
    )
    out.append("## Provenance\n")
    out.append(
        f"Config hash `{m.config_hash}`. Python {m.python}, numpy {m.numpy}. "
        f"Started {m.started}, finished {m.finished}. Files: `config.resolved.yaml`, `manifest.json`, "
        "`llm_calls.jsonl`, `shock_paths.parquet`, `trajectories.parquet`, `metrics.json`.\n"
    )
    path = d / "report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def write_experiment_report(exp_dir: str | Path) -> Path:
    exp = Path(exp_dir)
    index = pd.read_parquet(exp / "index.parquet")
    fig_dir = exp / "figures"
    fig_dir.mkdir(exist_ok=True)
    out = [f"# Experiment `{exp.name}`\n"]
    n_ok = int((index["status"] == "ok").sum())
    out.append(f"{len(index)} runs, {n_ok} ok, {len(index) - n_ok} failed.\n")

    out.append("## Runs\n")
    cols = ["run_id", "status", "model", "llm", "scenario", "prompt", "persona", "model_info",
            "answer", "intensity", "mode", "repeat", "n_llm_calls", "converged", "modified"]
    out.append(md_table(index.to_dict("records"), [c for c in cols if c in index.columns]))
    failed = index[index["status"] != "ok"]
    if len(failed):
        out.append("## Failures\n")
        out.append(md_table(failed.to_dict("records"), ["run_id", "error"]))

    summary = build_summary(run_table(exp)) if n_ok else pd.DataFrame()
    if not summary.empty:
        out.append("## Answers by cell (relative % change)\n")
        out.append("Raw LLM answers. `n` is the number of draws (1 at temperature 0).\n")
        answer_cols = ["llm", *AXES, "model_info", "n", "answer_mean", "answer_std", "answer_min", "answer_median", "answer_max"]
        out.append(md_table(summary.to_dict("records"), answer_cols))
        effect = info_effect(summary)
        if not effect.empty:
            out.append("## Effect of showing the model to the LLM\n")
            out.append("Mean answer with the model's name, description and parameters in the prompt minus without.\n")
            out.append(md_table(effect.to_dict("records"), ["llm", *AXES, "answer_without", "answer_with", "diff"]))
            out.append(f"Mean difference {effect['diff'].mean():.4g}; mean absolute difference "
                       f"{effect['diff'].abs().mean():.4g}.\n")

    runs: list[RunData] = [load_run(exp / rid) for rid in index.loc[index["status"] == "ok", "run_id"]]
    if runs:
        peak_rows, variables = [], []
        for r in runs:
            row = {"run_id": r.manifest.run_id}
            for v, vals in r.metrics.items():
                row[v] = vals["peak"]
                if v not in variables:
                    variables.append(v)
            peak_rows.append(row)
        out.append("## Peak response by run\n")
        out.append(md_table(peak_rows, ["run_id"] + variables))

        shown = variables[:12]
        style = load_style()
        seen_per_model: dict[str, int] = {}
        look = {}
        for r in runs:
            position = seen_per_model.get(r.manifest.model, 0)
            seen_per_model[r.manifest.model] = position + 1
            look[r.manifest.run_id] = (
                model_color(r.manifest.model, style),
                run_linestyle(position, style),
                run_alpha(position, style),
            )
        fig, axes = _grid(shown, "Impulse responses by run (colour = model)")
        for ax, v in zip(axes, shown):
            marked = set()
            for r in runs:
                d = r.trajectories[(r.trajectories["scenario"] == "irf") & (r.trajectories["variable"] == v)]
                if len(d):
                    c, ls, alpha = look[r.manifest.run_id]
                    ax.plot(d["t"], d["value"], label=run_label(r.manifest), color=c, linestyle=ls, alpha=alpha,
                            linewidth=1.8)
                    for period, shock_label in _shock_markers(r.shock_paths, r.config):
                        label = f"{shock_label} | {r.manifest.model}"
                        if (period, label) not in marked:
                            ax.axvline(period, color=style["event_color"], linestyle="--",
                                       linewidth=1, alpha=0.7, label=label)
                            marked.add((period, label))
            ax.axhline(0, color=style["neutral_color"], linestyle="--", linewidth=1, label="base-model")
            _style_axis(ax, v)
        for ax in axes[len(shown):]:
            ax.set_visible(False)
        _save(fig, fig_dir / "irf_overlay.png")
        out.append("## Overlay of impulse responses\n\n![overlay](figures/irf_overlay.png)\n")
    path = exp / "experiment_report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
