"""Plot colours.

Expect: every registered model has its own valid hex colour in configs/plotting.yaml, no model uses the neutral grey
reserved for baselines, unknown models get a deterministic fallback colour, and legend labels are short and unique.
"""

import re
from pathlib import Path

from matplotlib.colors import is_color_like

from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.reporting import report
from dgse_llm.reporting.style import (
    arm_label,
    arm_style,
    load_style,
    model_color,
    run_alpha,
    run_label,
    run_linestyle,
    shock_kinds,
)
from dgse_llm.schemas import RunManifest


def test_every_registered_model_has_a_unique_hex_colour():
    colors = load_style()["model_colors"]
    for name in MODEL_REGISTRY.names():
        assert name in colors, f"add '{name}' to model_colors in configs/plotting.yaml"
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", c) for c in colors.values())
    assert len({c.lower() for c in colors.values()}) == len(colors), "two models share a colour"


def test_all_configured_colours_are_valid_and_neutral_is_reserved():
    style = load_style()
    for c in [*style["model_colors"].values(), *style["fallback_palette"], style["neutral_color"]]:
        assert is_color_like(c), c
    assert style["neutral_color"].lower() not in {c.lower() for c in style["model_colors"].values()}


def test_font_sizes_are_configured_and_not_hardcoded():
    fonts = load_style()["fonts"]
    assert set(fonts) == {"suptitle", "title", "axis_label", "tick", "legend"}
    assert all(isinstance(v, (int, float)) and v > 0 for v in fonts.values())
    source = (Path(report.__file__)).read_text(encoding="utf-8")
    assert not re.search(r"fontsize\s*=\s*\d", source), "font sizes belong in configs/plotting.yaml"


def test_figure_size_is_configured_and_not_hardcoded():
    size = load_style()["figure"]
    assert set(size) == {"panel_width", "panel_height", "max_columns", "dpi"}
    assert all(v > 0 for v in size.values()) and int(size["max_columns"]) == size["max_columns"]
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(figsize|dpi)\s*=\s*[\d(]", source), "figure size and dpi belong in configs/plotting.yaml"


def test_known_model_uses_configured_colour():
    assert model_color("rbc") == load_style()["model_colors"]["rbc"]


def test_unknown_model_gets_deterministic_fallback():
    style = load_style()
    first = model_color("some_future_model")
    assert first == model_color("some_future_model") and first in style["fallback_palette"]


def test_linestyles_cycle():
    styles = load_style()["run_linestyles"]
    assert run_linestyle(0) == styles[0] and run_linestyle(len(styles)) == styles[0]
    assert len({run_linestyle(i) for i in range(len(styles))}) == len(styles)


def manifest(**kw):
    base = dict(run_id="r", experiment_id="e", config_hash="h", model="nk3", scenario="wedge")
    base.update(kw)
    return RunManifest(**base)


def test_run_label_includes_only_what_is_set():
    assert run_label(manifest()) == "nk3 | wedge"
    full = run_label(manifest(llm="mock", intensity=2.0, mode="replace", repeat=1))
    assert full == "nk3 | wedge | mock | i=2 | replace | r1"


def test_arm_labels():
    both = ["classic", "llm"]
    assert arm_label("baseline", both) == "base-model"
    assert arm_label("classic", both) == "base-model + classic shock"
    assert arm_label("llm", both) == "base-model + LLM shock"
    assert arm_label("shocked", both) == "base-model + classic shock + LLM shock"
    assert arm_label("shocked", ["llm"]) == "base-model + LLM shock"


def test_arm_styles_share_the_model_colour_and_differ_by_linestyle_and_shade():
    arms = ("baseline", "classic", "llm", "shocked")
    looks = {arm: arm_style(arm, "nk3") for arm in arms}
    assert {look["color"] for look in looks.values()} == {model_color("nk3")}, "one colour per model"
    assert len({look["linestyle"] for look in looks.values()}) == 4, "arms need different line styles"
    assert len({look["alpha"] for look in looks.values()}) == 4, "arms need different shades"
    assert len({look["linewidth"] for look in looks.values()}) == 1, "thick versus thin is hard to see"
    assert looks["shocked"]["alpha"] == 1.0
    assert looks["baseline"]["alpha"] < looks["classic"]["alpha"] < looks["llm"]["alpha"] < looks["shocked"]["alpha"]


def test_run_shades_cycle_and_start_opaque():
    alphas = load_style()["run_alphas"]
    assert alphas[0] == 1.0 and all(0 < a <= 1 for a in alphas)
    assert run_alpha(0) == 1.0 and run_alpha(len(alphas)) == 1.0
    assert len({run_alpha(i) for i in range(len(alphas))}) == len(alphas)


def test_shock_kinds_from_modified_variables():
    m = manifest(modified=[
        {"variable": "u", "mode": "additive", "source": "classic", "shock_id": "a"},
        {"variable": "wedge_pi", "mode": "additive", "source": "llm", "shock_id": "b"},
    ])
    assert shock_kinds(m) == ["classic", "llm"]
    assert shock_kinds(manifest()) == []
    only_manual = manifest(modified=[{"variable": "u", "mode": "additive", "source": "manual", "shock_id": "a"}])
    assert shock_kinds(only_manual) == ["classic"]
