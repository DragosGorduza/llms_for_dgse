# AGENTS.md

Guide for humans and coding agents working in this repo: the goal, the plan, the design logic, the conventions, and
the current status. Read this first.

## 1. Goal

Use an LLM to **modify one input of a standard macro model** (RBC, NK, Krusell-Smith, later TANK/HANK) instead of
replacing the model with LLM forecasts. The LLM output becomes a time path that is added to, or substituted for, an
exogenous input (for example the household taste shifter `d`). The macro model's own equations then produce the
consequences. Sources: [literature review](claude_lit_review_LLMs_for_DGSE.md), [planning notes](notes%20from%20planning.md),
and the Krusell-Smith notebook (`krusell_smith (1).ipynb`, the pandemic narrative driving `d`).

Why: standard macro models handle economic interactions well but not non-economic behaviour (regime shifts, fear,
news-driven expectation changes). The LLM supplies that behavioural input; the model keeps the economics and avoids
double counting by only touching a declared input variable. The lit review found no paper doing this inside a
NK/DSGE/HANK model.

An experiment can: modify one or several variables; apply a one-period shock (the model takes over afterwards) or a
multi-period path; add on top of the baseline input or replace it outright.

## 2. Plan (steps 0 to 6) and where each lives

| Step | Plan | Where |
| --- | --- | --- |
| 0 | File and folder structure, a README in each folder | whole repo; root [README.md](README.md) |
| 1 | Data structures for inputs (models, LLMs, shocks) and outputs (tracked variables, modified variables, impacts) | `src/dgse_llm/schemas/` |
| 2 | One `.py` and one `.md` per model, common harness, easy to add a model; solver independent of models | `src/dgse_llm/models/`, `src/dgse_llm/solvers/` |
| 3 | A harness to run any LLM on any prompt, easy to add an LLM | `src/dgse_llm/llms/`, `prompts/` |
| 4 | Shock generation: duration, intensity, target variable and model | `src/dgse_llm/shocks/` |
| 5 | Experiments combining everything, traces of every input and output, markdown reports | `src/dgse_llm/experiments/`, `src/dgse_llm/reporting/`, `configs/`, `runs/` |
| 6 | Tests everywhere, no secrets exposed | `tests/`, `secrets/` |

## 3. Architecture and the logic behind it

```
prompts/ template + narrative -> llms (client, cache, parser)
        -> shocks (mapping, time profile, intensity, mode) -> ShockPath
        -> experiments runner -> solver(model, modified inputs) -> run directory -> reporting
```

**Model = equations over whole time paths, no solver inside.** A model declares variables (inputs, unknowns, derived),
targets (equation residuals) and parameters, and implements `compute_steady_state`, `residuals(U, Z)` and `outputs`.
`U` are unknown paths, `Z` exogenous input paths, each a float64 array of shape `(T,)`. This is the "H(U, Z) = 0"
form of the SSJ tutorial. Because models never import a solver, switching solver changes only `solver:` in a config.

**Solver = finds U that zeroes the residuals.** `numpy_newton` computes the Jacobian once at the steady state by
finite differences and iterates quasi-Newton steps (as `solve_impulse_nonlinear` does in the notebook); `method:
linear` takes one step. `ssj` is an untested placeholder that needs a model with `to_ssj()`.

**LLM harness = provider-independent.** `LLMClient.complete()` checks a disk cache, then calls the provider. The cache
is only used at `temperature: 0`; at higher temperatures or with no temperature (reasoning models) every call is a
fresh draw and is never cached.
Providers: `openai` (Responses API) and `mock` (offline, regex rules; used in all default tests). The parser takes the
LAST JSON object in the answer, accepts single quotes and `%`, and raises `ParseError` (this replaces the notebook's
`get_dict`, which had an unbound-variable bug and broke on nested JSON). Keys are read lazily from the environment or
`secrets/secrets.env`; configs only hold the secret's NAME (`api_key_env`).

**Pydantic validates LLM types.** `LLMResponse` (provider output) is a pydantic model, and `parse_values` validates the
extracted JSON against a pydantic model built from the shock's `keys` (finite floats) and `answer_bounds` (plausible
range per key). Invalid answers are recorded with the error and fail the run before reaching the model.

**Shock = value x profile x intensity, injected per mode.**
- Value: manual (`intensity` in input units) or LLM (parsed number -> mapping function -> model units, times `intensity`).
- Profile: `one_time`, `multi_period`, `geometric_decay`, `from_series`, or `narrative_timeline` (a list of
  `{t, text}` events; one LLM call per distinct narrative, held until the next event, `tail_decay` after the last).
- Mode: `additive` (input = baseline + path) or `replace` (input = path where active, baseline elsewhere; for LLM
  shocks the mapping must output a level).
- **Classic shocks.** Each model defines ready-made shocks (`classic_shocks`: name, input, size, one-time /
  multi-period / decaying) in its `.py` and `.md`. A config uses one with `source: classic`, `classic: <name>`;
  `intensity` multiplies the size. They need no LLM and can be mixed with LLM shocks in one scenario.
- Only exogenous variables flagged `shockable` can be modified, and the resulting level must be inside the variable's
  `bounds` (error, or clip with `bounds_policy: clip`).

**Experiment = grid of cells, one run directory per cell.** Cells = models x scenarios x LLMs (only for scenarios with
an LLM shock) x `sweep.intensities` x `sweep.modes` x `sweep.repeats`. Shocks whose `model` differs from the cell's
model are dropped; a cell left with no shock is skipped. Each cell solves a baseline (cached per model, solver, `T`),
builds shock paths (every LLM call recorded), solves the shocked economy, and stores shocked minus baseline as the
impulse response (IRF). A failing cell is recorded as `failed`; the sweep continues.

**Classic + LLM together.** If a scenario has both classic/manual and LLM shocks, the cell solves four economies:
`baseline`, `classic` (without the LLM shock), `llm` (without the classic shock) and `shocked` (both), stores their
IRFs plus `irf_interaction` (combined minus the sum of the two; 0 in a linear model), and the report plots all four.
The LLM is called once, not once per economy.

**Outputs per run** (`runs/<experiment_id>/<run_id>/`): `config.resolved.yaml`, `manifest.json` (status, config hash,
modified variables, diagnostics, versions), `llm_calls.jsonl` (prompt, raw answer, parsed numbers, cache hit, no keys),
`shock_paths.parquet`, `trajectories.parquet` (baseline, shocked, irf; plus classic, llm, irf_classic, irf_llm,
irf_interaction when classic and LLM shocks are mixed), `metrics.json` (peak, time of peak,
cumulative, half-life; `metrics_arms.json` for the mixed case), and with `--report` `report.md` plus `figures/`. The experiment folder also has
`index.parquet` and `experiment_report.md`.

## 4. Modelling decisions worth knowing

- **Krusell-Smith `d`.** The notebook's backward step uses `+ d` in the envelope condition and `dplus` in the
  consumption rule, which is not one utility function. The port uses utility `c^(1-1/eis)/(1-1/eis) - d*c`, so
  marginal utility is `c^(-1/eis) - d`: positive `d` lowers consumption now. `eis` is constant; only `Z`, `d`, `beta`
  are shockable paths. Default grid is 7 x 100 (notebook: 500 assets) so tests are fast.
- **NK expectations wedge.** `wedge_pi` shifts inflation expectations in both the IS and Phillips curves. With
  `phi_pi * beta > 1` a positive one-period wedge raises inflation but LOWERS output on impact (closed form in
  `nk3.md`, checked in tests).
- **Perfect foresight, anticipated shocks.** The whole input path is known at t=0. Unanticipated sequential
  re-solving (notes, option 2) is not in v1; the intended hook is a `solve_from(t0)` method on the solver.
- **Lags before t=0 and leads after the horizon are the steady state.**

## 5. Conventions

- **Documentation is part of every model.** Each `Variable` has `units`, a plain-language `description`, `if_up` and
  `if_down`. The model's `.md` has one `### \`name\`` section per variable and target with **If it rises:** and
  **If it falls:** lines. Every function docstring states dtypes and shapes of inputs and outputs. These are
  enforced by `tests/models/test_conformance.py`.
- **Every code and config folder has a README.** Enforced by `tests/test_repo_hygiene.py`.
- **Registries and auto-discovery.** Models, solvers, LLM providers, profiles and mappings register with a decorator;
  non-underscore modules in `models/`, `solvers/` and `llms/providers/` are imported automatically.
- **Configs by name.** Experiment YAML refers to `configs/models|llms|shocks/<name>.yaml` by file stem; inline mappings
  also work.
- **Wiring lives in the shock config.** `configs/shocks/<name>.yaml` ties a prompt template (`llm.template`), the
  answer keys (`llm.keys`, `llm.key`, `llm.answer_bounds`), a mapping to model units (`llm.mapping`) and the model input
  (`model`, `variable`, `mode`) together. The LLM is chosen by the experiment (`llms:`), not the shock. The table is in
  the root README; keep it in sync when fields change.
- **Secrets.** Never in code, configs, prompts, notebooks or tests. A test scans for `sk-...` strings.
- **Plot colours.** One colour per macro model in `configs/plotting.yaml`, used by every figure; within a model,
  economies and runs differ by line style and shade (alpha), not by colour or width. Grey is only for zero reference
  lines. A new model must add its colour (enforced by `tests/reporting/test_style.py`).

## 6. How to add things

| To add | Do |
| --- | --- |
| A macro model | Copy `models/_template.py` to `<name>.py`, write `<name>.md` in the `rbc.md` format, add `configs/models/<name>.yaml` and a colour in `configs/plotting.yaml`; the conformance suite picks it up |
| An LLM provider | Copy `llms/providers/_template.py`; add `configs/llms/<name>.yaml` and the key name to `secrets/secrets.example.env` |
| A solver | New file in `solvers/` with `@register_solver`; select with `solver:` in the model config |
| A shock | New YAML in `configs/shocks/`; a new profile or mapping is one decorated function |
| An experiment | New YAML in `configs/experiments/` |

## 7. Working rules for agents in this repo

- **The user runs everything.** Do not install packages, create environments, or run code, tests or terminal
  commands. Write code carefully, check it statically, and tell the user which commands to run.
- Keep edits minimal and in the existing style; update the folder README and tests when behaviour changes.
- Never write, echo or commit a secret. If one is found, scrub it and tell the user to rotate it.
- The ipynb tools cannot be assumed to work on notebooks; verify any notebook edit by searching the raw file.

## 8. Status

Implemented: all of steps 0 to 6, three models (`rbc`, `nk3`, `krusell_smith`), solvers, OpenAI and mock providers,
shocks, runner, reports, CLI, example configs, and tests mirroring `src/`. **Nothing has been executed by the agent**;
the first test run is the first real check.

Commands (run by the user): `pip install -e ".[dev]"`, `pytest -m "not slow"`, `pytest`, `dgse-llm list`,
`dgse-llm run configs/experiments/smoke_manual.yaml --report`, `dgse-llm run configs/experiments/ks_pandemic_mock.yaml --report`.

Likely trouble spots on first run: the Krusell-Smith steady-state calibration (bracket `beta_lo`/`beta_hi`, tolerance,
runtime), finite-difference Jacobian noise for the heterogeneous-agent model, and pandas/pyarrow handling of empty frames.

Known limitations:
- **Re-running overwrites.** Run directories are named from the cell (no timestamp), so running the same command twice
  rewrites the same files. A failed re-run can leave stale `trajectories.parquet`, `metrics.json` and
  `shock_paths.parquet` from an earlier successful run next to a `failed` manifest. See section 9.
- Plain-numpy KS Jacobian costs about `T` household solves; use a small `T` while developing.
- `ssj` solver is a placeholder.

Not in v1: TANK and HANK models; unanticipated (sequential) shocks; decomposition of the LLM response into a
model-implied part and an LLM-only part (outputs already hold baseline, shocked and IRF so it can be added without
changing the layout); iterative LLM-model feedback loops.

## 9. Open questions and next steps

- Make reruns safe: delete the known output files of a run directory before writing, or add a timestamped run folder.
- First test run: fix whatever fails, starting with the `slow` Krusell-Smith tests.
- Decide how double counting is tested: LLM narrative that already contains the model's own forward-looking effect.
- Multi-model robustness: run the same scenario across several LLM backends (the backend acts as a parameter).
- Add TANK / HANK models and the unanticipated-shock solver.
