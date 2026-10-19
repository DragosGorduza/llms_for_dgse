"""Every YAML in configs/ must be valid and consistent with the code.

Expect: files validate against their schema, names refer to registered models/providers, shocks target shockable
inputs, templates exist and mention the keys the shock expects, and experiments expand into cells.
"""

import pytest

from dgse_llm.experiments import expand_cells, load_experiment
from dgse_llm.llms import LLM_REGISTRY
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import configs_dir, prompts_dir
from dgse_llm.schemas import LLMConfig, ModelConfig, ShockSpec, load_yaml
from dgse_llm.shocks import MAPPING_REGISTRY, check_variable, resolve_classic
from dgse_llm.solvers import SOLVER_REGISTRY


def files(kind):
    return sorted((configs_dir() / kind).glob("*.yaml"))


@pytest.mark.parametrize("path", files("models"), ids=lambda p: p.name)
def test_model_configs(path):
    cfg = ModelConfig.model_validate(load_yaml(path))
    assert cfg.name == path.stem and cfg.name in MODEL_REGISTRY
    assert cfg.solver in SOLVER_REGISTRY
    MODEL_REGISTRY.get(cfg.name)(cfg.params)


@pytest.mark.parametrize("path", files("llms"), ids=lambda p: p.name)
def test_llm_configs(path):
    cfg = LLMConfig.model_validate(load_yaml(path))
    assert cfg.name == path.stem and cfg.provider in LLM_REGISTRY


@pytest.mark.parametrize("path", files("shocks"), ids=lambda p: p.name)
def test_shock_configs(path):
    spec = ShockSpec.model_validate(load_yaml(path))
    assert spec.id == path.stem
    if spec.model:
        model = MODEL_REGISTRY.get(spec.model)()
        check_variable(resolve_classic(spec, model) if spec.source == "classic" else spec, model)
    if spec.llm:
        assert spec.llm.mapping in MAPPING_REGISTRY
        template = prompts_dir() / spec.llm.template
        assert template.exists()
        text = template.read_text(encoding="utf-8")
        for key in spec.llm.keys:
            assert key in text, f"{spec.llm.template} never asks for '{key}'"


@pytest.mark.parametrize("path", files("experiments"), ids=lambda p: p.name)
def test_experiment_configs(path):
    cfg = load_experiment(path)
    assert cfg.id == path.stem
    cells = expand_cells(cfg)
    assert cells, "experiment expands to no runs"
    for cell in cells:
        if cell.llm:
            assert cell.llm.provider in LLM_REGISTRY


def test_mock_experiments_do_not_need_secrets():
    for name in ("smoke_manual", "ks_pandemic_mock", "nk_wedge_mock", "nk_classic_vs_llm", "ks_classic_vs_llm"):
        cfg = load_experiment(configs_dir() / "experiments" / f"{name}.yaml")
        assert all(llm.provider == "mock" for llm in cfg.llms)


def test_expected_cell_counts():
    cell_count = lambda n: len(expand_cells(load_experiment(configs_dir() / "experiments" / f"{n}.yaml")))
    assert cell_count("smoke_manual") == 3
    assert cell_count("ks_pandemic_mock") == 3
    assert cell_count("nk_wedge_mock") == 2
    assert cell_count("nk_classic_vs_llm") == 1
    assert cell_count("ks_classic_vs_llm") == 1
