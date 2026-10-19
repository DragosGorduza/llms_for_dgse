# experiments/

Combines model, solver, LLM and shocks into runs and records everything.

| File | Role |
| --- | --- |
| `loader.py` | `load_experiment(path)`: reads YAML and resolves names to `configs/{models,llms,shocks}/<name>.yaml`. |
| `runner.py` | `expand_cells` builds the grid; `run_cell` runs one; `run_experiment` runs all and writes `index.parquet`. |
| `store.py` | `RunStore` writes a run directory; `load_run` reads it back. |
| `metrics.py` | Peak, time of peak, cumulative effect and half-life of each impulse response. |

## How names are resolved

`loader.py` reads the experiment YAML. Every string in `models`, `llms` and a scenario's `shocks` is a file stem that is
loaded from `configs/models/`, `configs/llms/` or `configs/shocks/`; the loaded content is validated against the
schemas in `src/dgse_llm/schemas/config.py`. The runner then receives fully resolved objects and writes them to each
run's `config.resolved.yaml`. The wiring of a prompt to a model variable lives in the shock config; see the root README.

## Grid

Cells = models x scenarios x LLMs (only for scenarios that contain an LLM shock) x `sweep.intensities` x
`sweep.modes` x `sweep.repeats`. Shocks whose `model` differs from the cell's model are dropped; if none are left the
cell is skipped. Repeats are separate cells (separate run directories); at temperature > 0 each repeat makes fresh
LLM calls because such answers are never cached.

## One cell

1. Build model and solver, solve the baseline (cached per model parameters, solver and `T`).
2. Build each shock path; every LLM call is recorded.
3. Inject the paths, solve again, take shocked minus baseline as the impulse response (IRF). If the scenario has both
   classic/manual and LLM shocks, also solve with only the classic ones and with only the LLM ones (no extra LLM
   calls), giving four economies: baseline, `classic`, `llm`, `shocked`, and the impulse responses `irf`, `irf_classic`,
   `irf_llm` and `irf_interaction` (combined minus the sum of the two separate effects).
4. Write config, shock paths, trajectories, metrics (and `metrics_arms.json` for the four-economy case), LLM calls
   and manifest.

A failing cell is recorded as `status: failed` with the error; the sweep continues.

## Outputs

See `runs/README.md`. Reports: `dgse-llm report <run_dir or experiment_dir>`.

## Tests (`tests/experiments`)

Cell expansion counts and skipping, end-to-end runs with the mock LLM on `rbc`, `nk3` and `krusell_smith`,
files and columns written, failure isolation, baseline reuse, loader name resolution, metrics on known IRFs.
