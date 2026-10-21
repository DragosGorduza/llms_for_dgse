# experiments/

Everything you edit to set up a test lives here. One experiment file describes the whole setup; the run command only
points at it: `dgse-llm run experiments/<model>/<name>.yaml --report` (or `dgse-llm run <model>/<name>`).

```
experiments/
  personas.yaml        who answers (id: text); experiments pick all or a subset
  narratives.yaml      optional broader context for a scenario (id: text)
  shocks/              one file per shock: just the event text, no timing (see shocks/README.md)
  <model>/
    README.md          the model write-up: what is shockable, what is not, which prompt shocks what
    model.json         register: description, every parameter, every variable (shockable or not)
    generic.yaml       the full grid for this model
    mock.yaml          small offline version with a mock LLM
```

Prompts live in `prompts/<model>/` (one `.j2` per prompt type plus `prompts.yaml`, which says which input each shocks).
Only `configs/plotting.yaml` (system-wide plot style) lives outside this folder.

## An experiment file

```yaml
id: ks_generic                    # tracks the experiment; also the output folder runs/<id>/
description: What this tests, in a sentence or two.
model: krusell_smith              # or {name: krusell_smith, params: {nA: 500}, solver: numpy_newton}

llms:                             # name -> settings; any number of entries, each run on every cell
  gpt4o_mini_t0: {provider: openai, model_id: gpt-4o-mini, temperature: 0.0, api_key_env: OPENAI_API_KEY}
  gpt4o_mini_t1: {provider: openai, model_id: gpt-4o-mini, temperature: 1.0, api_key_env: OPENAI_API_KEY}
  reasoner:      {provider: openai, model_id: o3-mini, temperature: null, api_key_env: OPENAI_API_KEY}

T: 120                            # periods simulated; long enough to see propagation
prompts: [tfp, discount_factor, household_consumption]   # prompts/krusell_smith/*.j2
personas: all                     # all, or a list of names from personas.yaml, or an inline mapping

scenarios:                        # name -> optional narrative and which shock happens in which period
  pandemic_one_time: {shocks: {4: pandemic_rumours}}
  mixed: {narrative: recession, shocks: {5: energy_price_spike, 12: sovereign_default, 20: technology_breakthrough}}

include_model_info: [false, true] # with and without the model's name, description and parameters in the prompt
repeats: 5                        # draws per cell when temperature is not 0
tracked: [Z, d, beta, Y, C, K, r, w]   # variables saved and reported (omit for all)
```

- **llms**: `temperature: 0.0` runs once and is cached; any other value, or `null` (models that take no temperature),
  runs `repeats` fresh draws per cell. `api_key_env` is the NAME of the secret in `secrets/secrets.env`, `null` if none.
  The same model can appear several times, for example at different temperatures. Extra provider settings (`seed`,
  `max_tokens`, `cache`, `options`) go in the same mapping.
- **scenarios**: `narrative` is a name from `narratives.yaml` or literal text; it is shown with every event. `shocks`
  maps periods to shock names (files in `shocks/`): `{5: A, 12: B, 20: C}` is shock A in period 5, B in period 12 and C
  in period 20. A key can be a range (`"8-11"`), a value can be a list (several shocks in one period), and the same
  shock can appear at several periods. Timing and combinations are set here and nowhere else. Each event is shown to
  the LLM in its own call and its answer is applied in that event's period(s) only.
- **include_model_info**: `[false, true]` runs every cell both ways; the report shows the difference.

Runs = LLMs x prompts x personas x scenarios x `include_model_info` (x `repeats` where temperature is not 0). Every
answer is a relative percentage change (`change_pct`), converted to the model input by `prompts/<model>/prompts.yaml`.

## Outputs

`runs/<id>/`: one folder per run, `index.parquet` (one row per run, with prompt, persona, scenario, model_info and the
raw answer), `summary.parquet` (answer distribution and mean peak responses per cell), `info_effect.parquet` (answer
with minus without model info) and `experiment_report.md`.

## Add

A new experiment is a new YAML in `experiments/<model>/`: copy `generic.yaml` and change what you need. A new persona
or narrative is one line in its library file; a new shock is a new file in `shocks/` with a `text:`. A new model needs `model.json`
here, `prompts/<model>/`, and the usual files in `src/dgse_llm/models/`.
