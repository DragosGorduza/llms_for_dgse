"""Loading experiment configs.

Expect: string entries resolve to files in configs/{models,llms,shocks}; inline mappings pass through; a missing
file raises FileNotFoundError.
"""

import pytest

from dgse_llm.experiments import load_experiment, resolve_experiment


def test_names_resolve_to_repo_configs():
    cfg = resolve_experiment(
        {
            "id": "x",
            "models": ["rbc"],
            "scenarios": [{"id": "s", "shocks": ["rbc_tfp"]}],
        }
    )
    assert cfg.models[0].name == "rbc"
    assert cfg.scenarios[0].shocks[0].variable == "Z"


def test_inline_entries_pass_through():
    cfg = resolve_experiment(
        {
            "id": "x",
            "models": [{"name": "nk3"}],
            "scenarios": [{"id": "s", "shocks": [{"id": "a", "variable": "u"}]}],
        }
    )
    assert cfg.models[0].name == "nk3" and cfg.scenarios[0].shocks[0].id == "a"


def test_missing_reference_raises():
    with pytest.raises(FileNotFoundError):
        resolve_experiment({"id": "x", "models": ["does_not_exist"], "scenarios": []})


def test_custom_configs_root(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mine.yaml").write_text("name: nk3\nparams: {kappa: 0.2}\n", encoding="utf-8")
    cfg = resolve_experiment(
        {"id": "x", "models": ["mine"], "scenarios": [{"id": "s", "shocks": [{"id": "a", "variable": "u"}]}]},
        configs_root=tmp_path,
    )
    assert cfg.models[0].params == {"kappa": 0.2}


def test_load_experiment_from_file(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_text("id: e\nmodels: [nk3]\nscenarios:\n  - id: s\n    shocks: [nk_cost_push]\n", encoding="utf-8")
    assert load_experiment(f).scenarios[0].shocks[0].id == "nk_cost_push"
