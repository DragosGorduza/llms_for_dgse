"""Prompt templates.

Expect: every template in prompts/ renders with the standard variables and contains the narrative; undefined
variables raise instead of silently producing an empty prompt.
"""

import jinja2
import pytest

from dgse_llm.llms import PromptRenderer
from dgse_llm.paths import prompts_dir

TEMPLATES = sorted(p.name for p in prompts_dir().glob("*.j2"))


def test_templates_exist():
    assert TEMPLATES


@pytest.mark.parametrize("name", TEMPLATES)
def test_template_renders_with_standard_variables(name):
    out = PromptRenderer().render(name, narrative="UNIQUE NARRATIVE", t=3)
    assert "UNIQUE NARRATIVE" in out
    assert "JSON" in out


def test_persona_can_be_overridden():
    out = PromptRenderer().render("household_consumption.j2", narrative="n", persona="You are a retiree.")
    assert out.startswith("You are a retiree.")


def test_undefined_variable_raises(tmp_path):
    (tmp_path / "bad.j2").write_text("hello {{ missing }}", encoding="utf-8")
    with pytest.raises(jinja2.UndefinedError):
        PromptRenderer(tmp_path).render("bad.j2")
