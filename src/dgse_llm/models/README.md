# models/

Macro models, one `.py` and one `.md` each. A model is a set of sequence-space equations `H(U, Z) = 0`
(`U` unknown paths, `Z` exogenous input paths) and knows nothing about solvers or LLMs.

| Model | Description |
| --- | --- |
| `rbc` | RBC with elastic labour: [rbc.md](rbc.md) |
| `nk3` | Three-equation NK with an expectations wedge: [nk3.md](nk3.md) |
| `krusell_smith` | Heterogeneous-agent Krusell-Smith, plain numpy: [krusell_smith.md](krusell_smith.md) |

## Contract (`base.py`)

- `variables`: tuple of `Variable(name, role, units, description, shockable, bounds, if_up, if_down)`. Roles: `exogenous`, `unknown`, `derived`. Only exogenous inputs can be `shockable`. `description` says what the variable is and its macro meaning; `if_up` / `if_down` say what it means when the variable is above / below its steady state.
- `targets`: names of equation residuals; there must be as many as unknowns.
- `classic_shocks`: tuple of `ClassicShock(name, variable, size, kind, description, duration, rho)`: at least three ready-made shocks per model, including one-time and multi-period ones, each described in plain language and listed in the model's `.md` under `## Classic shocks` (a table row and a `### Classic shock <name>` section each). Experiments use them with `source: classic`; see `../shocks/README.md`.
- `default_params`: every parameter a config may override.
- `compute_steady_state()`: scalar for every variable.
- `residuals(U, Z)`: dict of residual paths (length `T`), zero at the solution.
- `outputs(U, Z)`: all variable paths including derived ones.
- Use `lead(x, ss)` and `lag(x, ss)` for time shifts; leads past the horizon and lags before t=0 are the steady state.

## Add a model

1. Copy `_template.py` to `<name>.py` and edit. It is auto-discovered; nothing else to register.
2. Write `<name>.md` in the format of `rbc.md`: a plain-language intro, one `### \`name\`` section per variable and per target (with **If it rises:** and **If it falls:** lines for variables), parameters, and a function reference. The conformance tests check this.
3. Add a colour for the model in `configs/plotting.yaml` (`model_colors`), then `experiments/<name>/` (`README.md`, `model.json` register, `generic.yaml`, `mock.yaml`) and `prompts/<name>/` (`prompts.yaml` and one `.j2` per shockable input). Parameter overrides and the solver are set per experiment in its `model:` entry.
4. Run `pytest tests/models tests/experiments/test_register.py tests/llms/test_prompts.py`: the conformance, register and prompt suites pick the new model up automatically.

## Solver independence

The same model runs on any solver in `../solvers`. To support SSJ later, add an optional `to_ssj()` method; see
`../solvers/README.md`. Models never import a solver.

## Tests (`tests/models`)

`test_conformance.py` runs for every registered model: complete steady state, zero residuals at steady state,
zero shock gives zero response, docs mention every variable. Model-specific files check economic directions and
calibration targets.
