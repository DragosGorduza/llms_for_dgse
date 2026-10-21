"""Prompt templates (prompts/<model>/<type>.j2 and prompts.yaml).

Expect: every template renders from persona, narrative (optional context), event and model_info alone; model_info
appears only when given;
every template asks for the single key `change_pct` as a relative percentage; every prompts.yaml entry has a
template, a shockable input of that model and valid bounds; undefined variables raise instead of rendering empty.
"""

import jinja2
import pytest

from dgse_llm.experiments import load_prompt_table, load_register
from dgse_llm.llms import PromptRenderer
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import prompts_dir

TEMPLATES = sorted(
    str(p.relative_to(prompts_dir())).replace("\\", "/")
    for p in prompts_dir().glob("*/*.j2")
)


def test_templates_exist_for_every_model():
    for model in MODEL_REGISTRY.names():
        assert any(t.startswith(f"{model}/") for t in TEMPLATES), f"no prompts for model '{model}'"


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_renders_from_the_slots(name):
    out = PromptRenderer().render(
        name, persona="UNIQUE PERSONA", narrative="UNIQUE CONTEXT", event="UNIQUE EVENT", model_info=""
    )
    assert out.startswith("UNIQUE PERSONA") and "UNIQUE CONTEXT" in out and "UNIQUE EVENT" in out
    assert out.index("UNIQUE CONTEXT") < out.index("UNIQUE EVENT")
    assert '{"change_pct": <number>}' in out.strip().splitlines()[-1]
    assert "percentage change" in out and "JSON" in out


@pytest.mark.parametrize("name", TEMPLATES)
def test_model_info_appears_only_when_given(name):
    renderer = PromptRenderer()
    without = renderer.render(name, persona="p", narrative="n", event="e", model_info="")
    with_info = renderer.render(name, persona="p", narrative="n", event="e", model_info="MODEL BLOCK TEXT")
    assert "MODEL BLOCK TEXT" not in without and "MODEL BLOCK TEXT" in with_info


@pytest.mark.parametrize("name", TEMPLATES)
def test_prompts_never_ask_for_levels_or_percentage_points(name):
    out = PromptRenderer().render(name, persona="p", narrative="n", event="e", model_info="").lower()
    assert "percentage point" not in out


@pytest.mark.parametrize("model", MODEL_REGISTRY.names())
def test_prompt_table_matches_templates_and_register(model):
    table = load_prompt_table(model)
    register = load_register(model)
    assert table, f"{model} has no prompt types"
    on_disk = {p.stem for p in (prompts_dir() / model).glob("*.j2")}
    assert on_disk == set(table), "every template needs a prompts.yaml entry and vice versa"
    for kind, definition in table.items():
        var = register.variables[definition.variable]
        assert var.shockable, f"{model}/{kind}: {definition.variable} is not shockable"


@pytest.mark.parametrize("model", MODEL_REGISTRY.names())
def test_every_shockable_input_has_a_prompt(model):
    targeted = {d.variable for d in load_prompt_table(model).values()}
    shockable = {n for n, v in load_register(model).variables.items() if v.shockable}
    assert targeted == shockable


@pytest.mark.parametrize("model", MODEL_REGISTRY.names())
def test_zero_baseline_inputs_declare_a_conversion(model):
    register = load_register(model)
    for kind, definition in load_prompt_table(model).items():
        zero_baseline = "0 in the steady state" in register.variables[definition.variable].units
        assert (definition.per_pct is not None) == zero_baseline, f"{model}/{kind}: per_pct iff the baseline is 0"


def test_question_sentences_are_unique_within_a_model():
    for model in MODEL_REGISTRY.names():
        questions = []
        for kind in load_prompt_table(model):
            text = PromptRenderer().render(f"{model}/{kind}.j2", persona="p", narrative="n", event="e", model_info="")
            questions.append(next(line for line in text.splitlines() if line.startswith("Estimate the percentage change in")))
        assert len(set(questions)) == len(questions)


def test_persona_comes_first_and_missing_slot_raises():
    out = PromptRenderer().render("nk3/demand.j2", persona="You are a retiree.", narrative="n", event="e", model_info="")
    assert out.startswith("You are a retiree.")
    with pytest.raises(jinja2.UndefinedError):
        PromptRenderer().render("nk3/demand.j2", narrative="n", event="e", model_info="")


def test_narrative_context_is_optional():
    out = PromptRenderer().render("nk3/demand.j2", persona="p", narrative="", event="AN EVENT", model_info="")
    assert "AN EVENT" in out and out.startswith("p\nAN EVENT")


def test_undefined_variable_raises(tmp_path):
    (tmp_path / "bad.j2").write_text("hello {{ missing }}", encoding="utf-8")
    with pytest.raises(jinja2.UndefinedError):
        PromptRenderer(tmp_path).render("bad.j2")
