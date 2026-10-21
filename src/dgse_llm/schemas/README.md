# schemas/

Pydantic models for everything that is stored or configured. All round-trip through YAML/JSON and reject unknown keys.

## Inputs (`config.py`)

| Schema | Purpose |
| --- | --- |
| `ModelConfig` | Model `name`, parameter overrides, `solver` and `solver_options`; the `model:` entry of an experiment (a bare name works). |
| `LLMSetting`, `LLMConfig` | One entry of an experiment's `llms:` mapping: provider, model id, temperature (null if the model takes none), seed, `api_key_env` (the NAME of the secret, never the key; null if none), cache settings, provider `options`. `LLMConfig` adds the name (the mapping key). |
| `TimeProfile` | Shape of a shock over time: `at_periods` (listed `periods`), `one_time`, `multi_period`, `geometric_decay`, `from_series`, `narrative_timeline`. |
| `LLMSource` | How an LLM produces a shock value: prompt `template`, expected `keys`, `mapping`, `narrative` (context), `occurrences` (events), `scale_by_baseline` (answer is a % of the input's baseline), legacy `events`, `tail_decay`. |
| `Occurrence` | One event: `periods` and `text`; one LLM call, the answer applied in those periods only. |
| `ShockSpec` | One shock: target `variable` (and optional `model`), `mode` (`additive` or `replace`), `profile`, `intensity`, `source` (`manual`, `classic` = a named shock defined by the model, or `llm`). |
| `Scenario` | A named set of shocks applied together, plus `meta` (the experiment axes: prompt, persona, scenario, model_info). |
| `Sweep` | Intensities, modes and repeats to loop over; empty lists mean "use the shock's own value". |
| `ExperimentConfig` | The expanded plan the runner executes: models x LLMs x scenarios x sweep, horizon `T`, tracked variables. Built by the loader, not written by hand. |
| `ScenarioDef` | One scenario of an experiment: optional `narrative`, named `shocks`, and their resolved `events`. |
| `ExperimentSpec` | What a user writes in `experiments/<model>/<name>.yaml`: `id`, `description`, `model`, `llms` (name -> setting), `T` (default 120), `prompts`, `personas`, `scenarios`, `include_model_info` (list of bool), `repeats`, `tracked`. |
| `PromptDef` | One entry of `prompts/<model>/prompts.yaml`: the input a prompt shocks, `answer_bounds`, and `per_pct`. |
| `ModelRegister` (`ParamInfo`, `VarInfo`) | `experiments/<model>/model.json`: description, every parameter (default, meaning), every variable (role, shockable, units, meaning). |

## Outputs (`records.py`)

| Schema | Purpose |
| --- | --- |
| `LLMCall` | One LLM call: rendered prompt and hash, raw answer, parsed numbers, tokens, latency, cache hit, error. No keys. |
| `ShockPath` | The injected path per variable, with `mode`, `active` mask and provenance (LLM call ids). |
| `Diagnostics` | Solver convergence, iterations, residual norm. |
| `RunManifest` | Status, config hash, modified variables, the solved `arms`, solver diagnostics, versions, timestamps, `meta` (experiment axes) and `answer` (the raw LLM answer, relative % change). |

`stable_hash(obj)` gives a short content hash used for config hashes and prompt hashes.

## Tests (`tests/schemas`)

Defaults and validation errors (unknown keys, `llm` block mismatches, events without a narrative profile, missing
series), YAML round trip, hash stability.
