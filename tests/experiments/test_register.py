"""The per-model JSON register (experiments/<model>/model.json).

Expect: every registered model has a register that validates, covers every parameter and every variable of the model
code (shockable or not) with matching defaults, roles and shockable flags, and the model's experiments README names
all of them. Drift in either direction is reported.
"""

import pytest

from dgse_llm.experiments import describe_model, load_register, register_problems
from dgse_llm.experiments.register import register_path
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import experiments_dir

NAMES = MODEL_REGISTRY.names()


@pytest.mark.parametrize("model", NAMES)
def test_every_model_has_a_register_file(model):
    assert register_path(model).exists(), f"experiments/{model}/model.json is missing"


@pytest.mark.parametrize("model", NAMES)
def test_register_matches_the_model_code(model):
    register = load_register(model)
    assert register_problems(register, MODEL_REGISTRY.get(model)) == []


@pytest.mark.parametrize("model", NAMES)
def test_register_covers_every_variable_and_parameter(model):
    cls, register = MODEL_REGISTRY.get(model), load_register(model)
    assert set(register.variables) == {v.name for v in cls.variables}
    assert set(register.parameters) == set(cls.default_params)
    assert {n for n, v in register.variables.items() if v.shockable} == set(cls().shockable)
    assert all(v.role == "exogenous" for v in register.variables.values() if v.shockable)


@pytest.mark.parametrize("model", NAMES)
def test_experiments_readme_names_every_variable_and_parameter(model):
    text = (experiments_dir() / model / "README.md").read_text(encoding="utf-8")
    register = load_register(model)
    for name in [*register.variables, *register.parameters]:
        assert f"`{name}`" in text, f"experiments/{model}/README.md never mentions `{name}`"


@pytest.mark.parametrize("model", NAMES)
def test_every_model_folder_has_an_experiment(model):
    assert list((experiments_dir() / model).glob("*.yaml")), f"no experiment YAML for {model}"


def test_drift_is_detected():
    cls = MODEL_REGISTRY.get("nk3")
    register = load_register("nk3")
    missing_var = register.model_copy(update={"variables": {k: v for k, v in register.variables.items() if k != "pi"}})
    assert any("'pi'" in p for p in register_problems(missing_var, cls))
    wrong_flag = register.model_copy(update={"variables": {
        **register.variables, "u": register.variables["u"].model_copy(update={"shockable": False}),
    }})
    assert any("shockable" in p for p in register_problems(wrong_flag, cls))
    extra_param = register.model_copy(update={"parameters": {
        **register.parameters, "ghost": register.parameters["beta"],
    }})
    assert any("ghost" in p for p in register_problems(extra_param, cls))
    wrong_default = register.model_copy(update={"parameters": {
        **register.parameters, "beta": register.parameters["beta"].model_copy(update={"default": 0.5}),
    }})
    assert any("default" in p for p in register_problems(wrong_default, cls))


def test_describe_model_lists_parameters_and_the_shocked_variable_without_its_level():
    register = load_register("rbc")
    text = describe_model(register, {"alpha": 0.4}, "Z")
    assert register.title in text and "alpha = 0.4" in text and "sigma = 1" in text
    assert "`Z`" in text and register.variables["Z"].description in text
    assert register.variables["Z"].units not in text
