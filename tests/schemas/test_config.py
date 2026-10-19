"""Schemas validate inputs strictly and round-trip through YAML.

Expect: unknown keys, inconsistent llm/profile combinations and missing data raise ValidationError.
"""

import pytest
from pydantic import ValidationError

from dgse_llm.schemas import (
    ExperimentConfig,
    LLMConfig,
    LLMSource,
    ModelConfig,
    Scenario,
    ShockSpec,
    TimeProfile,
    dump_yaml,
    load_yaml,
    stable_hash,
)


def llm_block(**kw):
    base = dict(template="t.j2", keys=["a", "b"])
    base.update(kw)
    return LLMSource(**base)


def test_model_config_defaults():
    c = ModelConfig(name="rbc")
    assert c.solver == "numpy_newton" and c.params == {} and c.solver_options == {}


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(name="rbc", bogus=1)


def test_llm_config_allows_model_id_field():
    c = LLMConfig(name="x", provider="mock", model_id="m")
    assert c.model_id == "m" and c.api_key_env is None


def test_llm_source_key_defaults_to_first_and_validates():
    assert llm_block().key == "a"
    with pytest.raises(ValidationError):
        llm_block(key="zzz")


def test_llm_source_sorts_events():
    src = llm_block(events=[{"t": 5, "text": "b"}, {"t": 1, "text": "a"}])
    assert [e.t for e in src.events] == [1, 5]


def test_answer_bounds_must_name_keys_and_be_ordered():
    assert llm_block(answer_bounds={"a": [-10, 10]}).answer_bounds == {"a": (-10.0, 10.0)}
    with pytest.raises(ValidationError):
        llm_block(answer_bounds={"zzz": [0, 1]})
    with pytest.raises(ValidationError):
        llm_block(answer_bounds={"a": [5, 5]})
    with pytest.raises(ValidationError):
        llm_block(answer_bounds={"a": ["low", "high"]})


def test_manual_shock_with_llm_block_rejected():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="manual", llm=llm_block())


def test_llm_shock_without_block_rejected():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="llm")


def test_narrative_profile_needs_events_and_vice_versa():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="llm", llm=llm_block(), profile=TimeProfile(kind="narrative_timeline"))
    with pytest.raises(ValidationError):
        ShockSpec(
            id="s", variable="Z", source="llm",
            llm=llm_block(events=[{"t": 0, "text": "x"}]), profile=TimeProfile(kind="one_time"),
        )


def test_from_series_needs_series():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", profile=TimeProfile(kind="from_series"))


def test_classic_shock_needs_a_name_and_no_llm_block():
    spec = ShockSpec(id="s", source="classic", classic="tfp_slump_one_time")
    assert spec.variable is None and spec.profile is None and spec.intensity == 1.0
    with pytest.raises(ValidationError):
        ShockSpec(id="s", source="classic")
    with pytest.raises(ValidationError):
        ShockSpec(id="s", source="classic", classic="x", llm=llm_block())


def test_classic_name_only_allowed_for_classic_source():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="manual", classic="tfp_slump_one_time")


def test_manual_and_llm_shocks_still_need_a_variable():
    with pytest.raises(ValidationError):
        ShockSpec(id="s")
    with pytest.raises(ValidationError):
        ShockSpec(id="s", source="llm", llm=llm_block())


def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", mode="multiplicative")


def test_experiment_requires_models_and_scenarios():
    shock = ShockSpec(id="s", variable="Z")
    with pytest.raises(ValidationError):
        ExperimentConfig(id="e", models=[], scenarios=[Scenario(id="a", shocks=[shock])])
    with pytest.raises(ValidationError):
        ExperimentConfig(id="e", models=[ModelConfig(name="rbc")], scenarios=[])


def test_yaml_round_trip(tmp_path):
    cfg = ExperimentConfig(
        id="e",
        T=20,
        models=[ModelConfig(name="rbc")],
        scenarios=[Scenario(id="a", shocks=[ShockSpec(id="s", variable="Z", intensity=0.02)])],
    )
    path = tmp_path / "e.yaml"
    dump_yaml(cfg.model_dump(mode="json"), path)
    assert ExperimentConfig.model_validate(load_yaml(path)) == cfg


def test_stable_hash_is_deterministic_and_sensitive():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
