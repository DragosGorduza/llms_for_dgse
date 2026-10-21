# tests/

Mirrors `src/dgse_llm`. All default tests are offline and need no keys. Each test file states in its docstring what to expect.

| Path | Checks |
| --- | --- |
| `schemas/` | Validation rules, YAML round trip, hashing |
| `models/test_conformance.py` | Contract for every registered model (auto-includes new models) |
| `models/test_rbc.py`, `test_nk3.py`, `test_krusell_smith.py` | Economics: steady states, closed forms, signs, Walras' law |
| `solvers/` | Convergence, linear vs Newton, Jacobian cache, error handling, SSJ placeholder |
| `llms/` | Mock provider, disk cache, JSON parsing, secrets; `test_prompts.py` checks every `prompts/<model>/*.j2` and `prompts.yaml`; `test_live.py` is opt-in |
| `shocks/` | Profiles (incl. `at_periods`), mappings, additive/replace injection, bounds, relative and `per_pct` LLM shocks, narrative timelines |
| `experiments/` | `test_register.py`: every model has a `model.json` covering all variables and parameters; `test_loader.py`: spec compilation, grid, model info, repeats rule; `test_summary.py`: answer distributions, model-info effect; metrics, end-to-end runs, failure isolation, no secret leakage; `test_arms.py`: engine-level classic + LLM arms |
| `reporting/` | Report sections, figures, failed runs, experiment report |
| `evaluation/` | Like-with-like pairing, pair order and differences, repeats and z scores, axis importance, path distances, end-to-end `evaluations/` output and CLI |
| `test_configs.py` | `configs/` holds only `plotting.yaml`; every experiment and library in `experiments/` validates and matches the code (LLM settings, tracked variables, prompts, shocks, cell counts) |
| `test_cli.py` | `dgse-llm list`, `run`, `report` |
| `test_repo_hygiene.py` | READMEs exist (incl. `experiments/<model>/`), no key-like strings, gitignore rules, example secrets are blank |

## Run

- `pytest` runs everything offline.
- `pytest -m "not slow"` skips the Krusell-Smith tests (steady-state calibration and a T x T finite-difference Jacobian).
- `pytest --run-live` also runs tests that call real APIs; they skip themselves when no key is configured.
- `pytest tests/models/test_conformance.py` after adding a model.

## Helpers

`tests/conftest.py` has the `--run-live` flag and session fixtures that build each model once. `tests/helpers.py` has
`irf(model, T, deltas)` and `nk_experiment(...)`, a small experiment with manual and mock-LLM shocks.
