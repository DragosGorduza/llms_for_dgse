"""Plot style shared by all reports, read from configs/plotting.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dgse_llm.paths import configs_dir
from dgse_llm.schemas import RunManifest, load_yaml, stable_hash


@lru_cache(maxsize=None)
def _read(path: str) -> dict:
    return load_yaml(path)


def load_style(path: str | Path | None = None) -> dict:
    """The parsed plotting.yaml (keys model_colors, fallback_palette, run_linestyles, run_alphas, arms, neutral_color)."""
    return _read(str(path or configs_dir() / "plotting.yaml"))


def model_color(model: str, style: dict | None = None) -> str:
    """Hex colour of a model. Models missing from `model_colors` get a deterministic colour from `fallback_palette`."""
    style = style or load_style()
    colors = style["model_colors"]
    if model in colors:
        return colors[model]
    palette = style["fallback_palette"]
    return palette[int(stable_hash(model, 8), 16) % len(palette)]


def run_linestyle(position: int, style: dict | None = None) -> str:
    """Matplotlib line style for the position-th run of the same model (cycles through `run_linestyles`)."""
    styles = (style or load_style())["run_linestyles"]
    return styles[position % len(styles)]


def run_alpha(position: int, style: dict | None = None) -> float:
    """Shade (alpha) for the position-th run of the same model: 1 for the first, fainter for later ones; cycles."""
    alphas = (style or load_style())["run_alphas"]
    return alphas[position % len(alphas)]


def shock_kinds(m: RunManifest) -> list[str]:
    """Which kinds of shock a run applied: 'classic' (manual or model-defined) and/or 'llm', in that order."""
    kinds = []
    if any(x.get("source") != "llm" for x in m.modified):
        kinds.append("classic")
    if any(x.get("source") == "llm" for x in m.modified):
        kinds.append("llm")
    return kinds


def arm_label(arm: str, kinds: list[str]) -> str:
    """Legend text of an arm: baseline, classic, llm or shocked (everything applied; depends on `kinds`)."""
    names = {"classic": "classic shock", "llm": "LLM shock"}
    if arm == "baseline":
        return "base-model"
    if arm in names:
        return f"base-model + {names[arm]}"
    return "base-model + " + " + ".join(names[k] for k in kinds) if kinds else "scenario"


def arm_style(arm: str, model: str, style: dict | None = None) -> dict:
    """Matplotlib kwargs (color, linestyle, linewidth, alpha) for an arm.

    Every arm uses the model colour; arms differ by line style and by shade (alpha), as set under `arms` in plotting.yaml.
    """
    style = style or load_style()
    return {"color": model_color(model, style), **style["arms"][arm]}


def run_label(m: RunManifest) -> str:
    """Short legend label: model, scenario and whichever of LLM, intensity, mode, repeat are set."""
    parts = [m.model, m.scenario]
    if m.llm:
        parts.append(m.llm)
    if m.intensity is not None:
        parts.append(f"i={m.intensity:g}")
    if m.mode:
        parts.append(m.mode)
    if m.repeat:
        parts.append(f"r{m.repeat}")
    return " | ".join(parts)
