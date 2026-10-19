"""RBC economics.

Expect: textbook steady state, a temporary TFP boom raises output, consumption, investment and hours and then
fades, the resource constraint holds on every path, and an impatience shock lowers capital on impact.
"""

import numpy as np

from tests.helpers import impulse, irf

T = 200


def test_steady_state(models):
    m = models["rbc"]
    ss, p = m.steady_state, m.params
    assert abs(ss["r"] - (1 / p["beta"] - 1)) < 1e-12
    assert abs(ss["L"] - 1.0) < 1e-12
    assert abs(ss["Y"] - (ss["C"] + ss["I"])) < 1e-12
    assert abs(ss["I"] - p["delta"] * ss["K"]) < 1e-12


def test_tfp_shock_directions_and_decay(models):
    m = models["rbc"]
    z = 0.01 * 0.9 ** np.arange(T)
    base, shocked, d = irf(m, T, {"Z": z})
    for v in ("Y", "C", "I", "L"):
        assert d[v][0] > 0, v
    # Capital adjusts slowly, so check it has mostly returned rather than exactly zero.
    for v in ("Y", "K"):
        assert abs(d[v][-1]) < 0.05 * np.max(np.abs(d[v])), v


def test_resource_constraint_holds(models):
    m = models["rbc"]
    _, shocked, _ = irf(m, T, {"Z": impulse(T, 0.02)})
    p = shocked.paths
    assert np.max(np.abs(p["Y"] - p["C"] - p["I"])) < 1e-9


def test_impatience_lowers_capital_raises_consumption(models):
    m = models["rbc"]
    _, _, d = irf(m, T, {"beta": impulse(T, -0.005)})
    assert d["K"][0] < 0
    assert d["C"][0] > 0


def test_shocked_solution_converges(models):
    m = models["rbc"]
    shocked = irf(m, T, {"Z": impulse(T, 0.02)})[1]
    assert shocked.diagnostics.converged
