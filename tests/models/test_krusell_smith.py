"""Krusell-Smith model (plain numpy port).

Expect: the calibration targets are hit (r = 1%, Y = 1, assets = capital), a TFP boom raises output, wages and
consumption, a positive taste shifter d lowers consumption on impact and raises assets, and Walras' law holds.
Slow: solves a heterogeneous-agent steady state and a T x T Jacobian.
"""

import numpy as np
import pytest

from dgse_llm.models._het_agents import rouwenhorst
from tests.helpers import impulse, irf

pytestmark = pytest.mark.slow
T = 40


def test_calibration_targets(ks_model):
    ss = ks_model.steady_state
    assert abs(ss["r"] - 0.01) < 1e-12
    assert abs(ss["Y"] - 1.0) < 1e-12
    assert abs(ss["A"] - ss["K"]) < 1e-5
    assert 0.93 < ss["beta"] < 0.99


def test_income_process_has_unit_mean_and_valid_transitions():
    e, pi, Pi = rouwenhorst(0.966, 0.5, 7)
    assert abs(pi @ e - 1.0) < 1e-12
    assert np.allclose(Pi.sum(axis=1), 1.0)
    assert np.allclose(pi @ Pi, pi)


def test_tfp_shock(ks_model):
    z = 0.01 * 0.8 ** np.arange(T)
    base, shocked, d = irf(ks_model, T, {"Z": z})
    assert shocked.diagnostics.converged
    assert d["Y"][0] > 0 and d["w"][0] > 0 and d["C"][0] > 0
    assert d["K"][0] > 0


def test_taste_shifter_lowers_consumption_raises_assets(ks_model):
    _, shocked, d = irf(ks_model, T, {"d": impulse(T, 0.02)})
    assert shocked.diagnostics.converged
    assert d["C"][0] < 0
    assert d["K"][0] > 0


def test_walras_law(ks_model):
    _, shocked, _ = irf(ks_model, T, {"Z": 0.01 * 0.8 ** np.arange(T)})
    assert np.max(np.abs(shocked.paths["goods_mkt"])) < 1e-5


def test_impatience_shock_converges(ks_model):
    _, shocked, d = irf(ks_model, T, {"beta": impulse(T, 0.002)})
    assert shocked.diagnostics.converged
    assert d["K"][0] > 0
