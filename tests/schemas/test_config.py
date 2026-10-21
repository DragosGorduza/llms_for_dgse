"""Schemas validate inputs strictly and round-trip through YAML.

Expect: unknown keys, inconsistent llm/profile combinations and missing data raise ValidationError.
"""

import pytest
from pydantic import ValidationError

from dgse_llm.schemas import (
    ExperimentConfig,
    ExperimentSpec,
    LLMConfig,
    LLMSource,
    ModelConfig,
    ModelRegister,
    Occurrence,
    PromptDef,
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


def test_at_periods_profile_needs_periods():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", profile=TimeProfile(kind="at_periods"))
    assert ShockSpec(id="s", variable="Z", profile=TimeProfile(kind="at_periods", periods=[1, 4])).profile.periods == [1, 4]


def test_scale_by_baseline_needs_additive_mode():
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="llm", mode="replace", llm=llm_block(scale_by_baseline=True))


def event(*periods, text="something happens"):
    return {"periods": list(periods), "text": text}


def spec_kwargs(**over):
    base = dict(
        id="e", T=10, model="rbc", llms={"m": {"provider": "mock"}},
        prompts=["tfp"], personas={"p": "who"},
        scenarios={"s": {"shocks": ["x"], "events": [event(2)]}},
    )
    base.update(over)
    return base


def test_experiment_spec_defaults():
    spec = ExperimentSpec(**spec_kwargs())
    assert spec.include_model_info == [False, True] and spec.repeats == 5
    assert spec.model.name == "rbc" and spec.model.solver == "numpy_newton", "a bare model name works"


def test_experiment_spec_default_horizon_is_120():
    kwargs = spec_kwargs()
    del kwargs["T"]
    assert ExperimentSpec(**kwargs).T == 120


def test_experiment_spec_model_mapping_and_llm_settings():
    spec = ExperimentSpec(**spec_kwargs(
        model={"name": "nk3", "params": {"kappa": 0.2}},
        llms={
            "cold": {"provider": "openai", "model_id": "gpt-4o-mini", "temperature": 0.0, "api_key_env": "OPENAI_API_KEY"},
            "hot": {"provider": "openai", "model_id": "gpt-4o-mini", "temperature": 1.0, "api_key_env": "OPENAI_API_KEY"},
            "reasoner": {"provider": "openai", "model_id": "o3-mini", "temperature": None, "api_key_env": "OPENAI_API_KEY"},
        },
    ))
    assert spec.model.params == {"kappa": 0.2}
    assert [(k, v.temperature) for k, v in spec.llms.items()] == [("cold", 0.0), ("hot", 1.0), ("reasoner", None)]
    assert LLMConfig(name="cold", **spec.llms["cold"].model_dump()).api_key_env == "OPENAI_API_KEY"


@pytest.mark.parametrize("over", [
    dict(scenarios={"late": {"shocks": ["x"], "events": [event(10)]}}),
    dict(scenarios={"none": {"shocks": ["x"], "events": []}}),
    dict(scenarios={"noshock": {"events": [event(2)]}}),
    dict(scenarios={}),
    dict(include_model_info=[True, True]), dict(prompts=[]), dict(personas={}), dict(llms={}),
    dict(repeats=0), dict(bogus=1), dict(llms={"m": {"provider": "mock", "bogus": 1}}),
])
def test_experiment_spec_rejects_bad_input(over):
    with pytest.raises(ValidationError):
        ExperimentSpec(**spec_kwargs(**over))


def test_occurrence_needs_periods_that_are_not_negative():
    assert Occurrence(periods=[4, 5], text="x").periods == [4, 5]
    with pytest.raises(ValidationError):
        Occurrence(periods=[], text="x")
    with pytest.raises(ValidationError):
        Occurrence(periods=[-1], text="x")


def test_occurrences_and_timeline_events_cannot_be_mixed():
    src = llm_block(occurrences=[event(1)], events=[{"t": 0, "text": "x"}])
    with pytest.raises(ValidationError):
        ShockSpec(id="s", variable="Z", source="llm", llm=src, profile=TimeProfile(kind="narrative_timeline"))


def test_prompt_def_validates_bounds_and_conversion():
    assert PromptDef(variable="Z", answer_bounds=[-5, 5]).per_pct is None
    with pytest.raises(ValidationError):
        PromptDef(variable="Z", answer_bounds=[5, 5])
    with pytest.raises(ValidationError):
        PromptDef(variable="d", answer_bounds=[-5, 5], per_pct=0)


def test_model_register_rejects_missing_descriptions():
    ok = dict(
        name="m", title="A model", description="x" * 50,
        parameters={"a": {"default": 1.0, "description": "what a means"}},
        variables={"Z": {"role": "exogenous", "shockable": True, "units": "index", "description": "d" * 30}},
    )
    assert ModelRegister.model_validate(ok).variables["Z"].shockable
    bad = {**ok, "parameters": {"a": {"default": 1.0, "description": "x"}}}
    with pytest.raises(ValidationError):
        ModelRegister.model_validate(bad)
