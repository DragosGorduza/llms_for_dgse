# shocks/

Builds the path that modifies one exogenous input of a macro model, then injects it.

## Flow

`ShockSpec` + model (+ LLM) -> `build_shock_path` -> `ShockPath` -> `apply_shocks` -> modified exogenous paths.

1. **Check**: the variable must be an exogenous, `shockable` input of the model (`check_variable`).
2. **Value**: `source: manual` uses `intensity` itself. `source: classic` uses a ready-made shock defined by the model
   (see below); `intensity` multiplies its size. `source: llm` renders the prompt template, calls the LLM,
   parses `llm.keys`, takes `llm.key` and converts it with the mapping function; the result is multiplied by `intensity`.
3. **Time profile** (`profiles.py`, unit-height shape): `one_time` (impulse at `start`, the model's own dynamics
   take over afterwards), `multi_period` (`duration` periods), `geometric_decay` (`rho`), `from_series`.
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

One shock config wires one LLM answer to one model input. In `configs/shocks/ks_pandemic_d.yaml`:

1. `llm.template` names the file in `prompts/`. The code renders it with `narrative` (the event text), `t` (the
   period) and every entry of `llm.template_vars`.
2. The template asks for JSON with keys; `llm.keys` lists them. `llm.key` (default: the first) is the one that drives
   the shock. Other keys are validated but not used.
3. `llm.answer_bounds` rejects implausible raw answers (pydantic).
4. `llm.mapping` and `llm.mapping_params` convert the raw answer into the units of the model input. For example
   `linear` with `slope: -0.001` gives `d = -0.001 * social_change_pct`.
5. The value is multiplied by `intensity` and shaped by `profile` (and `llm.events`, `llm.tail_decay`).
6. The path is injected into the model input named by `variable`, in the given `mode`.

The link between an answer key and a variable is therefore `variable:` plus the single driving key; there is no
per-key variable setting. To drive several variables, use several shocks (each makes its own LLM call). Which LLM is
used is chosen by the experiment (`llms:`), not by the shock. The code is `_ask_llm` and `build_shock_path` in
`builder.py`; the full picture is in the root [README](../../../README.md).

## Classic shocks (no LLM)

Each model lists ready-made shocks in `classic_shocks` (in its `.py`, described in its `.md`): a name, the input it
hits, a size, and a timing (one-time, multi-period or decaying), for example `cost_push_energy` in `nk3`. Use one with

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

`intensity` is a multiplier on the mapped value (manual shocks: the value in input units). Experiments can sweep it
(`sweep.intensities`) without editing shock configs.

## Tests (`tests/shocks`)

Shape of each profile, additive vs replace arithmetic, bounds error/clip, non-shockable variable rejected, classic
shocks (size, timing, intensity multiplier, override, unknown name), LLM
shock with the mock provider (single call, narrative timeline, tail decay, one call per distinct narrative), parse
failures recorded and raised, provenance links to LLM call ids.
