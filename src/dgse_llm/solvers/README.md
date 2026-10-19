# solvers/

A solver finds the unknown paths that zero a model's residuals for given exogenous paths. Models and solvers
are independent: any solver runs any model that satisfies the `models/base.py` contract.

| Solver | Description |
| --- | --- |
| `numpy_newton` | Plain numpy/scipy. `method: newton` (default) iterates; `method: linear` takes one step. The Jacobian is computed once per (model parameters, T) by finite differences and cached. Options: `tol`, `maxit`, `fd_step`. |
| `ssj` | Placeholder adapter for `sequence_jacobian`. Needs a model with `to_ssj()`; untested. |

## Contract (`base.py`)

`solve(model, exog, T) -> Solution(paths, diagnostics)`. `exog` has one length-`T` array per exogenous variable;
`paths` has every model variable. `Diagnostics` reports `converged`, `iterations`, `residual_norm`.

## Switch solver

Set `solver: ssj` (and `solver_options`) in the model config. Nothing in models, shocks or experiments changes.

## Add a solver

Create `<name>.py` with a `Solver` subclass decorated with `@register_solver("<name>")`. It is auto-discovered.
Optional later extension: a `solve_from(t0, ...)` method for sequential, unanticipated shocks (not in v1).

## Tests (`tests/solvers`)

Convergence on each model, a zero shock stays at steady state, linear equals Newton on the linear NK model,
the Jacobian is cached, bad options are rejected, and the SSJ adapter refuses models without `to_ssj()`.
