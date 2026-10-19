# schemas/

Pydantic models for everything that is stored or configured. All round-trip through YAML/JSON and reject unknown keys.

## Inputs (`config.py`)

| Schema | Purpose |
| --- | --- |
| `ModelConfig` | Model `name`, parameter overrides, `solver` and `solver_options`. |
| `LLMConfig` | Provider, model id, temperature, seed, `api_key_env` (the NAME of the secret, never the key), cache settings, provider `options`. |
| `TimeProfile` | Shape of a shock over time: `one_time`, `multi_period`, `geometric_decay`, `from_series`, `narrative_timeline`. |
| `LLMSource` | How an LLM produces a shock value: prompt `template`, expected `keys`, `mapping`, `events`, `tail_decay`. |
| `ShockSpec` | One shock: target `variable` (and optional `model`), `mode` (`additive` or `replace`), `profile`, `intensity`, `source` (`manual`, `classic` = a named shock defined by the model, or `llm`). |
| `Scenario` | A named set of shocks applied together (one or several variables). |
| `Sweep` | Intensities, modes and repeats to loop over; empty lists mean "use the shock's own value". |
| `ExperimentConfig` | Models x LLMs x scenarios x sweep, horizon `T`, tracked variables. |

## Outputs (`records.py`)

| Schema | Purpose |
| --- | --- |
| `LLMCall` | One LLM call: rendered prompt and hash, raw answer, parsed numbers, tokens, latency, cache hit, error. No keys. |
| `ShockPath` | The injected path per variable, with `mode`, `active` mask and provenance (LLM call ids). |
| `Diagnostics` | Solver convergence, iterations, residual norm. |
| `RunManifest` | Status, config hash, modified variables, the solved `arms`, solver diagnostics, versions, timestamps. |

`stable_hash(obj)` gives a short content hash used for config hashes and prompt hashes.

## Tests (`tests/schemas`)

Defaults and validation errors (unknown keys, `llm` block mismatches, events without a narrative profile, missing
series), YAML round trip, hash stability.
