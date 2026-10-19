"""Adapter for the Sequence-Space Jacobian toolkit (sequence_jacobian). Untested placeholder for later use.

A model opts in by defining `to_ssj()` returning a dict with:
  model        : a sequence_jacobian model (create_model(...))
  calibration  : dict of calibrated parameters
  unknowns_ss, targets_ss : dicts for solve_steady_state
  unknowns, targets       : lists of names for the dynamic solve
The model's exogenous variable names must equal the SSJ input names.
"""

from __future__ import annotations

import numpy as np

from dgse_llm.models import Model
from dgse_llm.schemas import Diagnostics

from .base import Solution, Solver, register_solver


@register_solver("ssj")
class SSJSolver(Solver):
    name = "ssj"

    def __init__(self, maxit: int = 100, tol: float = 1e-8) -> None:
        super().__init__(maxit=maxit, tol=tol)
        self.maxit, self.tol = maxit, tol

    def solve(self, model: Model, exog: dict[str, np.ndarray], T: int) -> Solution:
        if not hasattr(model, "to_ssj"):
            raise NotImplementedError(f"Model '{model.name}' does not define to_ssj(); use solver 'numpy_newton'")
        spec = model.to_ssj()
        ssj_model = spec["model"]
        ss = ssj_model.solve_steady_state(
            spec["calibration"], spec["unknowns_ss"], spec["targets_ss"], solver="hybr"
        )
        shocks = {n: exog[n] - model.steady_state[n] for n in model.exogenous if np.any(exog[n] != model.steady_state[n])}
        td = ssj_model.solve_impulse_nonlinear(ss, spec["unknowns"], spec["targets"], shocks, maxit=self.maxit, tol=self.tol)
        paths = {k: np.asarray(ss[k] + td[k]) for k in td.keys() if k in ss}
        paths.update({n: exog[n] for n in model.exogenous})
        return Solution(paths, Diagnostics(solver=self.name, converged=True, iterations=0, residual_norm=0.0))
