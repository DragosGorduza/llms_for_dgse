"""Every experiment file and library in experiments/ must be valid and consistent with the code.

Expect: configs/ holds only plotting.yaml; every experiment sits in experiments/<model>/ for the model it names, lists
its LLMs with temperature and key reference, uses prompts, personas, narratives and shocks that exist, tracks
variables of that model, and expands into runs; the generic experiments are the full grid; mock experiments need no
secrets; the libraries (personas, narratives, shocks) are well formed.
"""

import pytest

from dgse_llm.experiments import expand_cells, load_experiment, load_prompt_table, load_register, resolve_spec
from dgse_llm.experiments.loader import load_shock_library
from dgse_llm.llms import LLM_REGISTRY
from dgse_llm.models import MODEL_REGISTRY
from dgse_llm.paths import configs_dir, experiments_dir
from dgse_llm.schemas import load_yaml
from dgse_llm.solvers import SOLVER_REGISTRY

EXPERIMENTS = sorted(experiments_dir().glob("*/*.yaml"))


def model_name(raw):
    return raw["model"] if isinstance(raw["model"], str) else raw["model"]["name"]


def test_configs_holds_only_the_plot_style():
    assert [p.name for p in configs_dir().rglob("*.yaml")] == ["plotting.yaml"], (
        "configs/models, llms, shocks and experiments moved into experiments/<model>/*.yaml; delete the old folders"
    )


@pytest.mark.parametrize("name", ["personas", "narratives"])
def test_libraries_are_non_empty_text(name):
    library = load_yaml(experiments_dir() / f"{name}.yaml")
    assert len(library) >= 3
    assert all(isinstance(v, str) and len(v) > 20 for v in library.values())


def test_shock_library_is_well_formed():
    library = load_shock_library()  # also rejects any file that carries timing
    assert len(library) >= 6
    assert all(len(text) > 20 for text in library.values())
    assert all((experiments_dir() / "shocks" / f"{name}.yaml").exists() for name in library)


def test_every_model_has_generic_and_mock_experiments():
    for model in MODEL_REGISTRY.names():
        assert (experiments_dir() / model / "generic.yaml").exists(), model
        assert (experiments_dir() / model / "mock.yaml").exists(), model


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_experiment_files(path):
    raw = load_yaml(path)
    assert model_name(raw) == path.parent.name, "an experiment lives in the folder of its model"
    assert raw["description"].strip(), "describe what the experiment tests"
    spec = resolve_spec(raw)
    assert spec.model.solver in SOLVER_REGISTRY
    assert set(raw["prompts"]) <= set(load_prompt_table(spec.model.name))
    assert spec.include_model_info
    variables = set(load_register(spec.model.name).variables)
    assert set(spec.tracked or variables) <= variables, "tracked names a variable the model does not have"
    for name, llm in spec.llms.items():
        assert llm.provider in LLM_REGISTRY, name
        assert "temperature" in raw["llms"][name], f"{name}: say which temperature (null if the model takes none)"
        assert "api_key_env" in raw["llms"][name] or llm.provider == "mock", f"{name}: give api_key_env or null"
    cells = expand_cells(load_experiment(path))
    assert cells, "experiment expands to no runs"
    assert "api_key:" not in path.read_text(encoding="utf-8")


def test_experiment_ids_are_unique():
    ids = [load_yaml(p)["id"] for p in EXPERIMENTS]
    assert len(ids) == len(set(ids))


def test_mock_experiments_do_not_need_secrets():
    for path in experiments_dir().glob("*/mock.yaml"):
        spec = resolve_spec(load_yaml(path))
        assert all(llm.provider == "mock" and llm.api_key_env is None for llm in spec.llms.values()), path


def test_generic_experiments_are_the_full_grid():
    for model in MODEL_REGISTRY.names():
        raw = load_yaml(experiments_dir() / model / "generic.yaml")
        spec = resolve_spec(raw)
        assert spec.T >= 120, "long horizon so propagation is visible"
        assert set(raw["prompts"]) == set(load_prompt_table(model)), model
        assert raw["personas"] == "all" and len(spec.personas) >= 4
        assert set(raw["include_model_info"]) == {True, False}
        assert raw["repeats"] >= 2 and raw.get("tracked")
        assert len(spec.llms) >= 2 and {llm.temperature for llm in spec.llms.values()} >= {0.0, 1.0}
        lengths = {len(s.events) for s in spec.scenarios.values()}
        assert 1 in lengths and max(lengths) > 1, "one-time and multi-period scenarios"
        assert any(s.narrative for s in spec.scenarios.values()), "some scenario with a broader narrative"
        assert any(len({o.name for o in s.events}) > 1 for s in spec.scenarios.values()), "different shocks over time"


def test_expected_cell_counts():
    def count(model, name):
        return len(expand_cells(load_experiment(experiments_dir() / model / f"{name}.yaml")))

    assert count("krusell_smith", "mock") == 3 * 2 * 2 * 2
    assert count("rbc", "mock") == 2 * 2 * 2 * 2
    assert count("nk3", "mock") == 4 * 2 * 2 * 2
    assert count("krusell_smith", "generic") == (3 * 4 * 6 * 2) * (1 + 5)
    assert count("nk3", "generic") == (4 * 4 * 6 * 2) * (1 + 5)
