# shocks/

Builds the path that modifies one exogenous input of a macro model, then injects it.

Users do not write `ShockSpec`s any more. An experiment file in `experiments/<model>/` is compiled by
`experiments/loader.py` into one single-LLM-shock `ShockSpec` per prompt x persona x scenario x model-info flag. This
package is the engine behind that. Manual, classic and narrative-timeline shocks remain available to code and tests,
not to experiment YAML.

## Flow

`ShockSpec` + model (+ LLM) -> `build_shock_path` -> `ShockPath` -> `apply_shocks` -> modified exogenous paths.

1. **Check**: the variable must be an exogenous, `shockable` input of the model (`check_variable`).
2. **Value**: `source: manual` uses `intensity` itself. `source: classic` uses a ready-made shock defined by the model
   (see below); `intensity` multiplies its size. `source: llm` renders the prompt template, calls the LLM,
   parses `llm.keys`, takes `llm.key` and converts it with the mapping function; the result is multiplied by `intensity`.
3. **Time profile** (`profiles.py`, unit-height shape): `at_periods` (1 in each listed period; one entry is a
   single-period shock; used by `build_shock_path` for a single call), `one_time` (impulse at `start`, the model's own
   dynamics take over afterwards), `multi_period` (`duration` periods), `geometric_decay` (`rho`), `from_series`.
   `narrative_timeline` is different: `llm.events` is a list of `{t, text}`; in each period the latest event's
   narrative is sent to the LLM (one call per distinct narrative) and its value is held until the next event.
   After the last event the value is multiplied by `tail_decay ** (t - t_last)`. Before the first event the shock is 0.
4. **Injection mode**:
   - `additive`: input = baseline + path (the path is a deviation in input units).
   - `replace`: input = path where the shock is active, baseline elsewhere (the path is a level in input units; for
     LLM shocks the mapping must output a level).
5. **Bounds**: the resulting input level must be inside the variable's `bounds`; otherwise `ShockError`, or clipped
   if `bounds_policy: clip`.

## Wiring: from prompt to model variable

For an experiment cell the compiled LLM shock is wired like this (`compile_spec` in `experiments/loader.py`, using
`prompts/<model>/prompts.yaml` and `experiments/<model>/model.json`):

1. `llm.template` is `<model>/<prompt>.j2`. It is rendered with `persona`, `model_info` (empty when the flag is off),
   `narrative` (the scenario's optional context, `llm.narrative`) and `event` (the event text of the call).
2. `llm.occurrences` lists the scenario's events, each with its periods. Each distinct event text is one LLM call
   whose answer is applied in that event's periods only (no holding between events; events in the same period add up).
3. The answer key is always `change_pct`, a relative percentage change.
4. `llm.answer_bounds` rejects implausible raw answers (pydantic), from `prompts.yaml`.
5. Conversion to model units: with `per_pct` in `prompts.yaml`, `mapping: linear` with that slope (inputs whose
   baseline is 0); without it, `mapping: pct` and `scale_by_baseline: true`, so the value is
   `change_pct / 100 x the input's baseline level` (checked in `build_shock_path`; a zero baseline raises `ShockError`).
6. The path is always injected `additive`, into the model input named by `variable`.

The link between the answer and a variable is `variable:` plus the single driving key. To drive several variables, use
several prompts (each makes its own LLM call). Which LLM is used is chosen by the experiment (`llms:`). The code is
`_ask_llm` and `build_shock_path` in `builder.py`; the full picture is in the root [README](../../../README.md).

## Classic shocks (no LLM, code-level only)

Each model lists ready-made shocks in `classic_shocks` (in its `.py`, described in its `.md`): a name, the input it
hits, a size, and a timing (one-time, multi-period or decaying), for example `cost_push_energy` in `nk3`. A
`ShockSpec` uses one with

```yaml
id: nk_classic_energy
model: nk3
source: classic
classic: cost_push_energy
intensity: 1.0      # multiplier of the model's size; -1 flips the sign
```

`variable` is taken from the model (if you give it, it must match) and `profile` from the model unless you set one to
override the timing. `dgse-llm list` prints every model's classic shocks.

## Classic and LLM shocks together

Put a classic (or manual) shock and an LLM shock in the same scenario. The runner then solves four economies and the
report plots them side by side:

| Economy (arm) | Shocks applied |
| --- | --- |
| `baseline` | none |
| `classic` | the classic/manual shocks only |
| `llm` | the LLM shocks only |
| `shocked` | all of them (classic + LLM) |

The LLM is called once, not once per economy. `irf_interaction` = combined response minus the sum of the two separate
responses: 0 in a linear model (`nk3`), and non-zero in a nonlinear one, which is one way to see double counting.
Scenarios with only one kind of shock keep just `baseline` and `shocked`.

## Mapping functions (`mapping.py`)

Before mapping, the raw LLM answer is validated by pydantic: finite number, inside `llm.answer_bounds` if given (for
example `answer_bounds: {social_change_pct: [-100, 100]}`). Invalid answers raise `ShockError` and are recorded.

Registered with `@register_mapping`: `identity`, `linear(slope, intercept)`, `pct(factor)` (value/100*factor).
Parameters come from `llm.mapping_params`. Add your own in one function.

## Intensity

`intensity` is a multiplier on the mapped value (manual shocks: the value in input units). It is not used by the
experiment files, which always apply the LLM's answer at full size.

## Tests (`tests/shocks`)

Shape of each profile (including `at_periods`), additive vs replace arithmetic, bounds error/clip, non-shockable
variable rejected, classic
shocks (size, timing, intensity multiplier, override, unknown name), LLM
shock with the mock provider (single call, narrative timeline, tail decay, one call per distinct narrative), relative
(`scale_by_baseline`) and `per_pct` conversions, parse
failures recorded and raised, provenance links to LLM call ids.
