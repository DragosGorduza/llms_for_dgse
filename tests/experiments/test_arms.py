"""Classic shock, LLM shock and both together (four economies).

Expect:
- a scenario with both a classic/manual and an LLM shock stores baseline, classic, llm and shocked (everything), plus
  the impulse responses irf, irf_classic, irf_llm and irf_interaction
- in a linear model (nk3) the combined response is exactly the sum of the two separate ones (interaction 0)
- scenarios with one kind of shock store only baseline and shocked, and write no arm metrics
- the LLM is called once per narrative, not once per arm
"""

import numpy as np
import pytest

from dgse_llm.experiments import load_run, run_experiment
from tests.helpers import nk_arms_experiment


@pytest.fixture(scope="module")
def arms_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("arms")
    manifests = run_experiment(nk_arms_experiment(), out_root=root)
    assert [m.status for m in manifests] == ["ok"], [m.error for m in manifests]
    return load_run(root / "arms" / manifests[0].run_id)


def values(run, scenario, variable):
    d = run.trajectories[(run.trajectories["scenario"] == scenario) & (run.trajectories["variable"] == variable)]
    return d.sort_values("t")["value"].to_numpy()


def test_all_four_economies_are_stored(arms_run):
    assert set(arms_run.trajectories["scenario"]) == {
        "baseline", "classic", "llm", "shocked", "irf", "irf_classic", "irf_llm", "irf_interaction",
    }
    assert arms_run.manifest.arms == ["baseline", "classic", "llm", "shocked"]
    assert set(arms_run.manifest.diagnostics) == {"baseline", "classic", "llm", "shocked"}
    assert all(d.converged for d in arms_run.manifest.diagnostics.values())


def test_shock_sources_are_recorded(arms_run):
    sources = {x["shock_id"]: x["source"] for x in arms_run.manifest.modified}
    assert sources == {"cp": "classic", "wedge": "llm"}


def test_each_economy_differs_from_baseline_in_the_expected_direction(arms_run):
    base = values(arms_run, "baseline", "pi")
    assert np.allclose(base, 0)
    assert values(arms_run, "irf_classic", "pi")[0] > 0, "cost-push raises inflation"
    assert values(arms_run, "irf_llm", "pi")[1] > 0, "the wedge starts at t=1 and raises inflation"
    assert np.allclose(values(arms_run, "classic", "pi"), values(arms_run, "irf_classic", "pi"))


def test_linear_model_has_no_interaction(arms_run):
    for v in ("pi", "x", "i"):
        total = values(arms_run, "irf", v)
        parts = values(arms_run, "irf_classic", v) + values(arms_run, "irf_llm", v)
        assert np.allclose(total, parts, atol=1e-10), v
        assert np.allclose(values(arms_run, "irf_interaction", v), 0, atol=1e-10), v


def test_arm_metrics_written_for_every_tracked_variable(arms_run):
    assert set(arms_run.arm_metrics) == {"classic", "llm", "classic_llm", "interaction"}
    for arm, per_variable in arms_run.arm_metrics.items():
        assert set(per_variable) == {"pi", "x", "i"}, arm
    assert arms_run.arm_metrics["classic_llm"]["pi"]["peak"] == arms_run.metrics["pi"]["peak"]


def test_llm_called_once_per_narrative_not_per_arm(arms_run):
    assert arms_run.manifest.n_llm_calls == 2 and len(arms_run.llm_calls) == 2


@pytest.mark.parametrize("kwargs, source", [({"classic_only": True}, "classic"), ({"llm_only": True}, "llm")])
def test_single_kind_scenarios_have_only_baseline_and_shocked(tmp_path, kwargs, source):
    manifest = run_experiment(nk_arms_experiment(**kwargs), out_root=tmp_path)[0]
    run = load_run(tmp_path / "arms" / manifest.run_id)
    assert manifest.status == "ok" and manifest.arms == ["baseline", "shocked"]
    assert set(run.trajectories["scenario"]) == {"baseline", "shocked", "irf"}
    assert run.arm_metrics == {} and not (run.run_dir / "metrics_arms.json").exists()
    assert run.manifest.modified[0]["source"] == source
