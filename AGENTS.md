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

An experiment can: modify one input per prompt type; run named scenarios, each an optional broader narrative plus one or
more named shocks placed at periods (`shocks: {5: A, 12: B, 20: C}`: a single entry is a one-time shock, several give a
multi-period path);
and be run with and without telling the LLM which model and parameters it is perturbing. Every LLM answer is a relative
percentage change, so answers are comparable across models, LLMs and prompts.

**Benchmarking goal (current focus).** Compare LLMs (and the same LLM across personas, scenarios and with or
without model information) as providers of the behavioural input: for each macro model, run the inner product of
scenarios x personas x relevant prompts, record every raw answer, and for non-deterministic LLMs (temperature other
than 0) repeat and record the distribution of answers and of model outcomes.

## 2. Plan (steps 0 to 6) and where each lives

| Step | Plan | Where |
| --- | --- | --- |
| 0 | File and folder structure, a README in each folder | whole repo; root [README.md](README.md) |
| 1 | Data structures for inputs (models, LLMs, experiment specs, model register) and outputs (tracked variables, modified variables, impacts) | `src/dgse_llm/schemas/` |
| 2 | One `.py` and one `.md` per model, plus a JSON register of its parameters and variables (shockable or not); common harness, easy to add a model; solver independent of models | `src/dgse_llm/models/`, `experiments/<model>/model.json`, `src/dgse_llm/solvers/` |
| 3 | A harness to run any LLM on any prompt, easy to add an LLM; prompts grouped per macro model, one file per prompt type, free slots (persona, narrative, event, model info) | `src/dgse_llm/llms/`, `prompts/<model>/` |
| 4 | Shocks and experiments merged in `experiments/`: per model a write-up, register and experiment files; shared personas, narratives and named shocks (event texts, no timing); always relative % changes; model settings and LLM settings live in the experiment file | `experiments/`, `src/dgse_llm/shocks/` |
| 5 | Experiments combining everything (inner product of prompts, personas, scenarios, model-info flag, LLMs), traces of every input and output, markdown reports; the run command only points at an experiment file | `src/dgse_llm/experiments/`, `src/dgse_llm/reporting/`, `runs/` (`configs/` holds only `plotting.yaml`) |
| 6 | Tests everywhere, no secrets exposed | `tests/`, `secrets/` |
| 7 | Benchmarking: repeats for non-deterministic LLMs, answer distributions, effect of showing the model to the LLM | `src/dgse_llm/experiments/summary.py`, `runs/<id>/summary.parquet`, `info_effect.parquet` |
| 8 | Evaluation: like-with-like comparisons of a run along each axis (model info, LLM, persona, scenario), tables and figures | `src/dgse_llm/evaluation/`, `runs/<id>/evaluations/`; `dgse-llm evaluate <id>` or `run --evaluate` |

## 3. Architecture and the logic behind it

```
experiments/<model>/NAME.yaml  (ExperimentSpec: id, description, model, llms, T, prompts, personas, scenarios,
        include_model_info, repeats, tracked) + experiments/personas.yaml, narratives.yaml, shocks/*.yaml
        -> experiments/loader.py: compile (prompts x personas x scenarios x model-info flag)
prompts/<model>/TYPE.j2 (persona, narrative, event, model_info) -> llms (client, cache, parser)
        -> shocks (relative % -> model units, events at their periods) -> ShockPath
        -> experiments runner -> solver(model, modified inputs) -> run directory -> summary + reporting
```

**Model register.** `experiments/<model>/model.json` holds the model's title and description, every parameter
(default and meaning) and every variable (role, shockable or not, units, meaning). It feeds the `model_info` block of a
prompt (`describe_model`) and is validated against the model code (`register_problems`) when loaded and in
`tests/experiments/test_register.py`: every model must have one, every parameter and variable must be covered, and the
shockable flags and defaults must match. The model's experiments README must also mention every variable and parameter.

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

**Shock = relative % change x periods, always additive.**
- Unit: every prompt asks for a relative percentage change and answers `{"change_pct": <number>}`; prompts never state the
  variable's current level. `prompts/<model>/prompts.yaml` converts it: `per_pct` (input units per +1%, for inputs with
  baseline 0 such as KS `d` and every `nk3` input) or, without it, `baseline x change_pct / 100` (`scale_by_baseline`).
- Timing: a shock is only an event text, one file in `experiments/shocks/` with no timing. A scenario in the experiment
  file maps periods to shocks (`shocks: {5: A, "8-11": B, 12: [C, D]}`) and has an optional broader `narrative`. Each
  distinct event text is one LLM call and its value is applied in that event's period(s) only (nothing is held between
  events; same-period values add). One entry is a one-time shock, several give a multi-period path. (`llm.occurrences`.)
- Because answers are relative changes, `additive` and `replace` coincide for these shocks; experiments use `additive`.
- Only exogenous variables flagged `shockable` can be modified, and the resulting level must be inside the variable's
  `bounds` (error, or clip with `bounds_policy: clip`).
- Engine-only extras that experiment files do not use: manual and classic shocks (`classic_shocks` in each model),
  `narrative_timeline`, `replace` mode, `intensity`.

**Experiment = grid of cells, one run directory per cell.** Cells = LLMs x prompts x personas x scenarios x
`include_model_info` (a list of bools: `[false, true]` runs both) x draws. Draws = `repeats` only when the LLM's
temperature is not 0 (or is null, as for reasoning models); at temperature 0 the answer is deterministic and cached, so
one draw. Each cell solves a baseline (cached per model, solver, `T`), builds the shock path (the LLM call is recorded),
solves the shocked economy, and stores shocked minus baseline as the impulse response (IRF). A failing cell is recorded
as `failed`; the sweep continues.

**Benchmark outputs.** After a sweep: `index.parquet` (one row per run with its axes and raw `answer`), `summary.parquet`
(per llm/prompt/narrative/persona/timing/model_info: n, answer mean/std/min/median/max, mean and std of peak responses)
and `info_effect.parquet` (answer with minus without model info). The experiment report prints both tables.

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
  **If it falls:** lines. Every function docstring states dtypes and shapes of inputs and outputs. The model also has
  `experiments/<model>/model.json` (register) and `experiments/<model>/README.md` (what is shockable, what is not).
  These are enforced by `tests/models/test_conformance.py` and `tests/experiments/test_register.py`.
- **Every code and config folder has a README.** Enforced by `tests/test_repo_hygiene.py`.
- **Registries and auto-discovery.** Models, solvers, LLM providers, profiles and mappings register with a decorator;
  non-underscore modules in `models/`, `solvers/` and `llms/providers/` are imported automatically.
- **Configs live in the experiment.** `configs/` holds only `plotting.yaml` (system-wide). An experiment file carries
  the model (name, parameter overrides, solver) and the `llms:` mapping (name -> provider, model id, temperature or
  null, `api_key_env` or null, options); personas, narratives and shocks come from `experiments/personas.yaml`,
  `narratives.yaml` and `shocks/`. The run command points at one experiment file (`dgse-llm run <model>/<name>`).
- **Wiring lives in `prompts/<model>/prompts.yaml`.** It ties each prompt type to the input it shocks (`variable`), the
  plausible answer range (`answer_bounds`) and the unit conversion (`per_pct`). The LLMs, personas, scenarios
  and model-info flag are chosen by the experiment. The table is in the root README; keep it in sync when fields change.
- **Prompt rules.** Four slots (`persona`, `narrative` optional context, `event`, `model_info`); end with `_answer_format.j2`; one unique
  question sentence per prompt type (the mock LLM matches it); never state the variable's level; always relative %.
- **Secrets.** Never in code, configs, prompts, notebooks or tests. A test scans for `sk-...` strings.
- **Plot colours.** One colour per macro model in `configs/plotting.yaml`, used by every figure; within a model,
  economies and runs differ by line style and shade (alpha), not by colour or width. Grey is only for zero reference
  lines. A new model must add its colour (enforced by `tests/reporting/test_style.py`).

## 6. How to add things

| To add | Do |
| --- | --- |
| A macro model | Copy `models/_template.py` to `<name>.py`, write `<name>.md` in the `rbc.md` format, add a colour in `configs/plotting.yaml`, then `experiments/<name>/` (`README.md`, `model.json`, `generic.yaml`, `mock.yaml`) and `prompts/<name>/` (`prompts.yaml` + one `.j2` per shockable input); the conformance, register and prompt suites pick it up |
| An LLM provider | Copy `llms/providers/_template.py`; add the key name to `secrets/secrets.example.env`; use it in an experiment's `llms:` entry |
| A solver | New file in `solvers/` with `@register_solver`; select with `solver:` in the model config |
| A prompt type | `prompts/<model>/<type>.j2` plus an entry in `prompts/<model>/prompts.yaml` |
| A narrative or persona | One line in `experiments/narratives.yaml` or `experiments/personas.yaml` |
| A shock | A new file `experiments/shocks/<name>.yaml` with a `text:` (no timing); reference it from a scenario's `shocks: {period: name}` |
| An experiment | New YAML in `experiments/<model>/` |

## 7. Working rules for agents in this repo

- **The user runs everything.** Do not install packages, create environments, or run code, tests or terminal
  commands. Write code carefully, check it statically, and tell the user which commands to run.
- Keep edits minimal and in the existing style; update the folder README and tests when behaviour changes.
- Never write, echo or commit a secret. If one is found, scrub it and tell the user to rotate it.
- The ipynb tools cannot be assumed to work on notebooks; verify any notebook edit by searching the raw file.

## 8. Status

Implemented: steps 0 to 7 in code, three models (`rbc`, `nk3`, `krusell_smith`) each with a register, prompts and
experiment files, solvers, OpenAI and mock providers, runner, summaries, reports, CLI, and tests mirroring `src/`.
**Nothing has been executed by the agent**; the first test run is the first real check. The refactor from
`configs/models|llms|shocks|experiments` to `experiments/<model>/` + `prompts/<model>/` is also unexecuted.

Commands (run by the user): `pip install -e ".[dev]"`, `pytest -m "not slow"`, `pytest`, `dgse-llm list`,
`dgse-llm run experiments/nk3/mock.yaml --report`, `dgse-llm run experiments/rbc/mock.yaml --report`,
`dgse-llm run experiments/krusell_smith/mock.yaml --report` (slow).

Likely trouble spots on first run: the Krusell-Smith steady-state calibration (bracket `beta_lo`/`beta_hi`, tolerance,
runtime; the KS mock experiment has 48 cells), finite-difference Jacobian noise for the heterogeneous-agent model,
pandas/pyarrow handling of empty frames, the `per_pct` calibrations in `prompts/nk3/prompts.yaml` (reasoned, not
checked), and `tests/reporting/test_report.py::test_plots_show_orange_baseline`, which predates the current plot
signatures and is likely stale.

Known limitations:
- **Re-running.** Runs that finished `ok` with an unchanged setup (same config hash, same rendered prompts) are kept and
  skipped; failed or changed runs are cleared and run again; `--force` reruns all (`runner._reusable`). A failed run is
  logged under `runs/<id>/errors/<ErrorName>/<run_id>.json` with the prompt and the returned text.
- Plain-numpy KS Jacobian costs about `T` household solves; use a small `T` while developing.
- `ssj` solver is a placeholder.
- The KS taste shifter `d` is one path for all households; a shock reaching only some (for example wealthy) households
  needs a household-model change, not a prompt change.
- Engine leftovers not reachable from experiment files: manual and classic shocks, `narrative_timeline`, `replace`
  mode, classic + LLM arms. Delete them if they are not wanted (tests in `tests/shocks`, `tests/experiments/test_arms.py`).

Not in v1: TANK and HANK models; unanticipated (sequential) shocks; decomposition of the LLM response into a
model-implied part and an LLM-only part; iterative LLM-model feedback loops.

## 9. Open questions and next steps

- First test run: fix whatever fails, starting with `pytest -m "not slow"`, then the `slow` Krusell-Smith tests.
- Review the `per_pct` calibrations and `answer_bounds` in each `prompts/<model>/prompts.yaml`.
- Reruns are safe now; open: reruns do not notice a code change in a model or solver (only the setup and the prompts).
- Decide how double counting is tested: LLM narrative that already contains the model's own forward-looking effect.
- Benchmark analysis: compare answers and outcomes across LLMs (the backend acts as a parameter), personas and
  narratives; test whether model information moves answers (`info_effect.parquet`).
- Add TANK / HANK models and the unanticipated-shock solver; add an exposure weight to KS `d` for subgroup shocks.
