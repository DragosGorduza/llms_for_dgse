"""Three-equation NK economics.

Expect: impact responses match the closed form for a one-period shock (after t=0 everything is back at the
steady state), the equations hold on solved paths, and responses scale linearly.
With k = 1 + phi_pi*kappa + phi_x and sigma = 1:
  monetary shock e:   x_0 = -e / k,            pi_0 = kappa * x_0
  cost-push shock u:  x_0 = -phi_pi * u / k,   pi_0 = u + kappa * x_0
  natural rate r_n:   x_0 = r_n / k,           pi_0 = kappa * x_0
  expectations wedge w: x_0 = w (1 - phi_pi*beta) / k,   pi_0 = beta*w + kappa * x_0
"""

import numpy as np

from tests.helpers import impulse, irf

T = 60


def _k(m):
    p = m.params
    return 1 + p["phi_pi"] * p["kappa"] + p["phi_x"]


def test_monetary_tightening(models):
    m = models["nk3"]
    _, _, d = irf(m, T, {"eps_i": impulse(T, 0.0025)})
    assert np.isclose(d["x"][0], -0.0025 / _k(m), atol=1e-10)
    assert np.isclose(d["pi"][0], m.params["kappa"] * d["x"][0], atol=1e-10)


def test_cost_push(models):
    m = models["nk3"]
    _, _, d = irf(m, T, {"u": impulse(T, 0.002)})
    assert np.isclose(d["x"][0], -m.params["phi_pi"] * 0.002 / _k(m), atol=1e-10)
    assert d["pi"][0] > 0


def test_natural_rate(models):
    m = models["nk3"]
    _, _, d = irf(m, T, {"r_n": impulse(T, 0.002)})
    assert np.isclose(d["x"][0], 0.002 / _k(m), atol=1e-10)
    assert d["pi"][0] > 0


def test_expectations_wedge(models):
    m = models["nk3"]
    p, w = m.params, 0.005
    _, _, d = irf(m, T, {"wedge_pi": impulse(T, w)})
    x0 = w * (1 - p["phi_pi"] * p["beta"]) / _k(m)
    assert np.isclose(d["x"][0], x0, atol=1e-10)
    assert np.isclose(d["pi"][0], p["beta"] * w + p["kappa"] * x0, atol=1e-10)
    assert d["pi"][0] > 0


def test_one_period_shock_leaves_no_echo(models):
    _, _, d = irf(models["nk3"], T, {"u": impulse(T, 0.003)})
    assert np.max(np.abs(d["pi"][1:])) < 1e-10 and np.max(np.abs(d["x"][1:])) < 1e-10


def test_equations_hold_on_solved_paths(models):
    m = models["nk3"]
    _, shocked, _ = irf(m, T, {"u": impulse(T, 0.003), "wedge_pi": 0.001 * 0.8 ** np.arange(T)})
    p = shocked.paths
    res = m.residuals({k: p[k] for k in m.unknowns}, {k: p[k] for k in m.exogenous})
    for r in res.values():
        assert np.max(np.abs(r)) < 1e-10


def test_linearity_scales_responses(models):
    _, _, d1 = irf(models["nk3"], T, {"u": 0.001 * 0.8 ** np.arange(T)})
    _, _, d2 = irf(models["nk3"], T, {"u": 0.002 * 0.8 ** np.arange(T)})
    assert np.allclose(d2["pi"], 2 * d1["pi"], atol=1e-10)
