"""The per-model JSON register (experiments/<model>/model.json): description, parameters, variables.

It is the single place that says what a model is, which inputs may be shocked and what every parameter means. It
feeds the 'model info' block of a prompt, and tests check it against the model code so it cannot drift.
"""

from __future__ import annotations

from pathlib import Path

from dgse_llm.paths import experiments_dir
from dgse_llm.schemas import ModelRegister


def register_path(model: str, root: Path | None = None) -> Path:
    return Path(root or experiments_dir()) / model / "model.json"


def register_problems(register: ModelRegister, model_cls) -> list[str]:
    """Differences between a register and the model class (empty list = consistent).

    Checks: same name; every default parameter present with the same default and no extra ones; every variable present
    with the same role and shockable flag and no extra ones.
    """
    problems: list[str] = []
    if register.name != model_cls.name:
        problems.append(f"name '{register.name}' differs from the model's '{model_cls.name}'")
    code_params = model_cls.default_params
    for name in sorted(set(code_params) - set(register.parameters)):
        problems.append(f"parameter '{name}' is not in the register")
    for name in sorted(set(register.parameters) - set(code_params)):
        problems.append(f"register parameter '{name}' does not exist in the model")
    for name in sorted(set(code_params) & set(register.parameters)):
        if float(register.parameters[name].default) != float(code_params[name]):
            problems.append(
                f"parameter '{name}': register default {register.parameters[name].default} != code {code_params[name]}"
            )
    code_vars = {v.name: v for v in model_cls.variables}
    for name in sorted(set(code_vars) - set(register.variables)):
        problems.append(f"variable '{name}' is not in the register")
    for name in sorted(set(register.variables) - set(code_vars)):
        problems.append(f"register variable '{name}' does not exist in the model")
    for name in sorted(set(code_vars) & set(register.variables)):
        reg, var = register.variables[name], code_vars[name]
        if reg.role != var.role:
            problems.append(f"variable '{name}': register role '{reg.role}' != code '{var.role}'")
        if reg.shockable != var.shockable:
            problems.append(f"variable '{name}': register shockable {reg.shockable} != code {var.shockable}")
    return problems


def load_register(model: str, root: Path | None = None) -> ModelRegister:
    """Read and validate experiments/<model>/model.json, and check it against the registered model class."""
    from dgse_llm.models import MODEL_REGISTRY

    path = register_path(model, root)
    if not path.exists():
        raise FileNotFoundError(f"No model register for '{model}': expected {path}")
    register = ModelRegister.model_validate_json(path.read_text(encoding="utf-8"))
    problems = register_problems(register, MODEL_REGISTRY.get(model))
    if problems:
        raise ValueError(f"{path.name} of '{model}' does not match the model code: " + "; ".join(problems))
    return register


def describe_model(register: ModelRegister, params: dict[str, float], variable: str) -> str:
    """The 'model info' text shown to the LLM when include_model_info is on.

    Contains the model's name and description, every parameter with the value used and its meaning, and what the
    shocked variable is. Deliberately omits the variable's current level: answers are always relative changes.
    """
    lines = [
        f"The question concerns the economic model: {register.title}.",
        register.description,
        "Parameters of the model (value used, meaning):",
    ]
    for name, info in register.parameters.items():
        value = params.get(name, info.default)
        lines.append(f"- {name} = {value:g}: {info.description}")
    lines.append(f"The quantity that your answer will change in the model is `{variable}`: "
                 f"{register.variables[variable].description}")
    return "\n".join(lines)
