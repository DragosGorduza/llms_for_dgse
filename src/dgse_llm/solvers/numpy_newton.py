"""Quasi-Newton perfect-foresight solver in plain numpy/scipy.

The Jacobian of the residuals is computed once at the steady state by finite differences and reused for every
Newton step, as in the SSJ tutorial's `solve_impulse_nonlinear`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from dgse_llm.models import Model
from dgse_llm.schemas import Diagnostics

from .base import Solution, Solver, register_solver

_JAC_CACHE: dict[tuple, Any] = {}


def clear_jacobian_cache() -> None:
    _JAC_CACHE.clear()


@register_solver("numpy_newton")
class NumpyNewton(Solver):
    """method='newton' iterates to tolerance; method='linear' takes a single step from the steady state."""

    name = "numpy_newton"

    def __init__(self, method: str = "newton", tol: float = 1e-8, maxit: int = 50, fd_step: float = 1e-6) -> None:
        super().__init__(method=method, tol=tol, maxit=maxit, fd_step=fd_step)
        if method not in ("newton", "linear"):
            raise ValueError("method must be 'newton' or 'linear'")
        self.method, self.tol, self.fd_step = method, tol, fd_step
        self.maxit = 1 if method == "linear" else maxit

    @staticmethod
    def _unpack(model: Model, x: np.ndarray, T: int) -> dict[str, np.ndarray]:
        return {n: x[i * T : (i + 1) * T] for i, n in enumerate(model.unknowns)}

    def _H(self, model: Model, x: np.ndarray, exog: dict[str, np.ndarray], T: int) -> np.ndarray:
        res = model.residuals(self._unpack(model, x, T), exog)
        return np.concatenate([np.asarray(res[n], dtype=float) for n in model.targets])

    def _x_ss(self, model: Model, T: int) -> np.ndarray:
        return np.concatenate([np.full(T, model.steady_state[n]) for n in model.unknowns])

    def _jacobian_lu(self, model: Model, T: int):
        key = (model.cache_key(), T, self.fd_step)
        if key not in _JAC_CACHE:
            exog = model.baseline_exog(T)
            x0 = self._x_ss(model, T)
            f0 = self._H(model, x0, exog, T)
            if f0.size != x0.size:
                raise ValueError(
                    f"Model '{model.name}' has {x0.size // T} unknowns but {f0.size // T} targets"
                )
            J = np.empty((f0.size, x0.size))
            for j in range(x0.size):
                h = self.fd_step * max(1.0, abs(x0[j]))
                x = x0.copy()
                x[j] += h
                J[:, j] = (self._H(model, x, exog, T) - f0) / h
            _JAC_CACHE[key] = lu_factor(J)
        return _JAC_CACHE[key]

    def solve(self, model: Model, exog: dict[str, np.ndarray], T: int) -> Solution:
        for n in model.exogenous:
            if n not in exog or len(exog[n]) != T:
                raise ValueError(f"exogenous path '{n}' missing or not of length T={T}")
        lu = self._jacobian_lu(model, T)
        x = self._x_ss(model, T)
        r = self._H(model, x, exog, T)
        norm = float(np.max(np.abs(r)))
        iterations = 0
        while norm > self.tol and iterations < self.maxit and np.isfinite(norm):
            x = x - lu_solve(lu, r)
            iterations += 1
            r = self._H(model, x, exog, T)
            norm = float(np.max(np.abs(r)))
        converged = bool(np.isfinite(norm) and norm <= self.tol)
        paths = model.outputs(self._unpack(model, x, T), exog)
        paths = {k: np.asarray(v, dtype=float).copy() for k, v in paths.items()}
        return Solution(paths, Diagnostics(solver=self.name, converged=converged, iterations=iterations, residual_norm=norm))
