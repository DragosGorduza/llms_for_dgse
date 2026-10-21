# experiments/ (package)

Turns an experiment file into runs and records everything. The user-facing files live in the top-level
[experiments/](../../../experiments/README.md) folder, one subfolder per macro model.

| File | Role |
| --- | --- |
| `loader.py` | `load_experiment(path)`: reads `experiments/<model>/<name>.yaml` (an `ExperimentSpec`), resolves personas, narratives and shock texts from `experiments/`, and `compile_spec` expands prompts x personas x scenarios x model-info flag into scenarios. |
| `register.py` | `load_register(model)` reads `experiments/<model>/model.json` and checks it against the model code; `describe_model` renders the model-info block of a prompt. |
| `runner.py` | `expand_cells` builds the grid; `run_cell` runs one; `run_experiment` runs all, then writes `index.parquet` and `summary.parquet`. |
| `summary.py` | Answer distribution per cell, and the effect of showing the model to the LLM (`summary.parquet`, `info_effect.parquet`). |
| `store.py` | `RunStore` writes a run directory; `load_run` reads it back; `write_index`. |
| `metrics.py` | Peak, time of peak, cumulative effect and half-life of each impulse response. |

## How a file becomes runs

1. `resolve_spec`: `personas` are names from `experiments/personas.yaml`, `all`, or an inline mapping. Each scenario's
   `shocks` map periods to shock names (`{5: A, 12: B}`; keys may be inclusive ranges like `"4-12"`, values lists).
   A range applies the event's single LLM response at every listed period. Each name is a file
   `experiments/shocks/<name>.yaml` holding only the event `text`. The result is the scenario's `events`. A
   scenario's `narrative` is a name from `narratives.yaml` or
   literal text. `model` (a name or `{name, params, solver}`) and `llms` (name -> setting) are read as written.
2. `compile_spec`: for each prompt, read its entry in `prompts/<model>/prompts.yaml`, check the input is shockable in
   `model.json`, and make one `Scenario` (one LLM `ShockSpec` with `occurrences`) per persona x scenario x model-info
   flag. The model-info text comes from `describe_model` and is passed as the template variable `model_info` (empty
   when off).
3. `expand_cells`: scenarios x LLMs x draws. **Draws = `repeats` only when the LLM's temperature is not 0** (including
   no temperature, as for reasoning models); at temperature 0 the answer is deterministic and cached, so one draw.

## One cell

1. Build model and solver, solve the baseline (cached per model parameters, solver and `T`).
2. Build the shock path (one LLM call per distinct event, recorded), inject it, solve again, take shocked minus baseline
   as the impulse response (IRF).
3. Write config, shock paths, trajectories, metrics, LLM calls and manifest (with the axes and the raw `answer`).

A failing cell is recorded as `status: failed` with the error and logged under `errors/<ErrorName>/<run_id>.json`
(prompt and returned text included); the sweep continues. `run_experiment` first checks every cell with `_reusable`: a
run that finished ok with the same config hash and the same rendered prompts is kept untouched (`on_skip`); anything
else is cleared (`RunStore.clear`) and run again; `force=True` / `--force` reruns all. The engine can still solve
classic + LLM combinations (`classic`, `llm`, `irf_interaction` arms) when handed such an `ExperimentConfig` in code;
experiment YAML does not produce them.

## Outputs

See `runs/README.md`. Reports: `dgse-llm report <run_dir or experiment_dir>`; the experiment report adds the answer
distribution and the with/without model-info table.

## Tests (`tests/experiments`)

Spec compilation (grid, wiring, model info, repeats rule), the register against every model, answer summaries,
end-to-end mock runs on `nk3` and `krusell_smith`, files and columns written, failure isolation, baseline reuse,
metrics on known IRFs.
