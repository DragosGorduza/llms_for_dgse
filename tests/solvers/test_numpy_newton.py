"""NumpyNewton solver.

Expect: zero shock needs no iterations, shocked problems converge below tolerance, the linear method equals Newton on
a linear model and is close on a mildly nonlinear one, the Jacobian is computed once per (model, T), and bad inputs
raise clear errors.
"""

import numpy as np
import pytest

from dgse_llm.solvers import SOLVER_REGISTRY, build_solver
from dgse_llm.solvers import numpy_newton as nn
from tests.helpers import impulse, irf


def test_registry_lists_solvers():
    assert {"numpy_newton", "ssj"} <= set(SOLVER_REGISTRY.names())
    with pytest.raises(KeyError, match="Unknown solver"):
        build_solver("nope")


def test_zero_shock_needs_no_iterations(models):
    m = models["rbc"]
    sol = build_solver("numpy_newton").solve(m, m.baseline_exog(30), 30)
    assert sol.diagnostics.converged and sol.diagnostics.iterations == 0


def test_shock_converges_below_tolerance(models):
    m = models["rbc"]
    _, shocked, _ = irf(m, 60, {"Z": impulse(60, 0.05)})
    d = shocked.diagnostics
    assert d.converged and d.iterations >= 1 and d.residual_norm <= 1e-8


def test_linear_equals_newton_on_linear_model(models):
    m = models["nk3"]
    delta = {"u": 0.002 * 0.7 ** np.arange(40)}
    _, newton, _ = irf(m, 40, delta, build_solver("numpy_newton", {"method": "newton"}))
    _, linear, _ = irf(m, 40, delta, build_solver("numpy_newton", {"method": "linear"}))
    for k in newton.paths:
        assert np.allclose(newton.paths[k], linear.paths[k], atol=1e-10), k
    assert linear.diagnostics.converged


def test_linear_close_to_newton_for_small_nonlinear_shock(models):
    m = models["rbc"]
    delta = {"Z": 1e-4 * 0.9 ** np.arange(80)}
    _, newton, _ = irf(m, 80, delta, build_solver("numpy_newton"))
    _, linear, _ = irf(m, 80, delta, build_solver("numpy_newton", {"method": "linear"}))
    assert np.allclose(newton.paths["C"], linear.paths["C"], atol=1e-6)


def test_nonconvergence_is_reported_not_hidden(models):
    m = models["rbc"]
    solver = build_solver("numpy_newton", {"tol": 1e-30, "maxit": 2})
    _, shocked, _ = irf(m, 40, {"Z": impulse(40, 0.05)}, solver)
    assert not shocked.diagnostics.converged
    assert shocked.diagnostics.iterations == 2


def test_jacobian_is_cached_per_model_and_horizon(models):
    m = models["rbc"]
    nn.clear_jacobian_cache()
    solver = build_solver("numpy_newton")
    solver.solve(m, m.baseline_exog(25), 25)
    assert len(nn._JAC_CACHE) == 1
    solver.solve(m, m.baseline_exog(25), 25)
    assert len(nn._JAC_CACHE) == 1
    solver.solve(m, m.baseline_exog(26), 26)
    assert len(nn._JAC_CACHE) == 2


def test_bad_inputs_rejected(models):
    m = models["rbc"]
    with pytest.raises(ValueError):
        build_solver("numpy_newton", {"method": "bogus"})
    with pytest.raises(ValueError, match="length"):
        build_solver("numpy_newton").solve(m, m.baseline_exog(10), 12)
