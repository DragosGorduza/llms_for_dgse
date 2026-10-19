# tests/

Mirrors `src/dgse_llm`. All default tests are offline and need no keys. Each test file states in its docstring what to expect.

| Path | Checks |
| --- | --- |
| `schemas/` | Validation rules, YAML round trip, hashing |
| `models/test_conformance.py` | Contract for every registered model (auto-includes new models) |
| `models/test_rbc.py`, `test_nk3.py`, `test_krusell_smith.py` | Economics: steady states, closed forms, signs, Walras' law |
| `solvers/` | Convergence, linear vs Newton, Jacobian cache, error handling, SSJ placeholder |
| `llms/` | Mock provider, disk cache, JSON parsing, secrets, prompts; `test_live.py` is opt-in |
| `shocks/` | Profiles, mappings, additive/replace injection, bounds, LLM shocks and narrative timelines |
| `experiments/` | Metrics, config loading, grid expansion, end-to-end runs, failure isolation, no secret leakage; `test_arms.py`: classic + LLM shocks give four economies and a zero interaction in a linear model |
| `reporting/` | Report sections, figures, failed runs, experiment report |
| `test_configs.py` | Every YAML in `configs/` validates and matches the code |
| `test_cli.py` | `dgse-llm list`, `run`, `report` |
| `test_repo_hygiene.py` | READMEs exist, no key-like strings, gitignore rules, example secrets are blank |

## Run

- `pytest` runs everything offline.
- `pytest -m "not slow"` skips the Krusell-Smith tests (steady-state calibration and a T x T finite-difference Jacobian).
- `pytest --run-live` also runs tests that call real APIs; they skip themselves when no key is configured.
- `pytest tests/models/test_conformance.py` after adding a model.

## Helpers

`tests/conftest.py` has the `--run-live` flag and session fixtures that build each model once. `tests/helpers.py` has
`irf(model, T, deltas)` and `nk_experiment(...)`, a small experiment with manual and mock-LLM shocks.
