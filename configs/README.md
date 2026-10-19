# configs/

YAML inputs. An experiment refers to model, LLM and shock configs by file name (without `.yaml`); inline
mappings work too. Schemas are in `src/dgse_llm/schemas/config.py`.

| Folder | Schema | Files |
| --- | --- | --- |
| `models/` | `ModelConfig` | `rbc`, `nk3`, `krusell_smith` |
| `llms/` | `LLMConfig` | `mock_pandemic`, `mock_expectations` (offline), `openai_gpt4o_mini` |
| `shocks/` | `ShockSpec` | `rbc_tfp`, `ks_tfp`, `nk_cost_push` (manual); `nk_classic_energy`, `ks_classic_precaution` (classic, defined by the model); `ks_pandemic_d`, `nk_wedge_energy` (LLM) |
| `experiments/` | `ExperimentConfig` | `smoke_manual`, `ks_pandemic_mock`, `nk_wedge_mock`, `nk_classic_vs_llm`, `ks_classic_vs_llm` (classic + LLM, four economies), `ks_pandemic_openai` |

## How files refer to each other

- An experiment lists `models`, `llms` and each scenario's `shocks` by file stem. `experiments/loader.py` replaces each
  name with the content of `configs/models/<name>.yaml`, `configs/llms/<name>.yaml` or `configs/shocks/<name>.yaml`
  (inline mappings also work) and validates the result against the schemas.
- A shock config refers to a prompt template (`llm.template` -> `prompts/`), a mapping function (`llm.mapping` ->
  `src/dgse_llm/shocks/mapping.py`) and a model input (`model` + `variable` -> the model's `.py` and `.md`).
- A shock config does not name the LLM; the experiment's `llms` list does.
- Example: `ks_pandemic_d` is read by `ks_pandemic_mock`, `ks_pandemic_openai` and `ks_classic_vs_llm`.

The field-by-field wiring of a prompt to a model variable is in the root README and `src/dgse_llm/shocks/README.md`.

## Conventions

- `plotting.yaml` (top level of this folder) sets one colour per macro model for all figures; add a colour there when you add a model. See `src/dgse_llm/reporting/README.md`.
- `shock.model` restricts a shock to one model; cells whose model has no matching shock are skipped.
- `intensity` is in input units for manual shocks and a multiplier on the mapped answer for LLM shocks.
- `sweep.intensities` and `sweep.modes` override every shock's own values; `sweep.repeats` repeats stochastic LLM calls.
- Never put a key in any config; `api_key_env` is the name of the secret.

## Run

`dgse-llm run configs/experiments/smoke_manual.yaml --report` (or `python -m dgse_llm ...`). Outputs go to `runs/<id>/`.

## Add

A new config is a new file here; every file is validated by `tests/test_configs.py`.

## Expect

`smoke_manual` finishes with all runs ok. `ks_pandemic_mock` writes 3 runs (intensities 0.5, 1, 2) with 6 LLM calls each.
`nk_wedge_mock` writes 2 runs with 2 calls each.
