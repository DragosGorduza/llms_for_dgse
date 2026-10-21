# prompts/

Jinja2 templates rendered by `dgse_llm.llms.PromptRenderer`. One folder per macro model, one `.j2` per prompt type.

```
prompts/
  _answer_format.j2        shared last paragraph: relative % change, JSON key `change_pct`
  <model>/
    prompts.yaml           prompt type -> model input it shocks, answer range, unit conversion
    <type>.j2              e.g. tfp.j2, discount_factor.j2, household_consumption.j2
```

Every template has exactly four degrees of freedom, all filled by the experiment (`experiments/<model>/*.yaml`):
`persona` (who answers), `narrative` (the scenario's optional broader context, empty if none), `event` (what happens in
the period being asked about, from the scenario's shocks) and `model_info` (the model's name, description and
parameters, rendered from `experiments/<model>/model.json`; empty when `include_model_info` is off, so templates wrap
it in `{% if model_info %}`). Undefined variables raise an error.

| Model | Prompt types (input shocked) |
| --- | --- |
| `krusell_smith` | `tfp` (`Z`), `discount_factor` (`beta`), `household_consumption` (`d`) |
| `rbc` | `tfp` (`Z`), `discount_factor` (`beta`) |
| `nk3` | `demand` (`r_n`), `cost_push` (`u`), `monetary_policy` (`eps_i`), `inflation_expectations` (`wedge_pi`) |

## One unit everywhere

Every prompt asks for a relative percentage change and the answer is always `{"change_pct": <number>}`: +5 is 5% higher
than otherwise, -5 is 5% lower. Prompts never state the current level of the variable. `prompts.yaml` converts the
answer into the model input:

| Field | Meaning |
| --- | --- |
| `variable` | The shockable input that moves (checked against `experiments/<model>/model.json`). |
| `answer_bounds` | Plausible range of the answer in percent; anything outside is rejected as invalid. |
| `per_pct` | Input units per +1% answered, for inputs whose baseline is 0 (`d` and every `nk3` input). |
| (no `per_pct`) | The answer is a relative change of the input's own baseline level: `Z` and `beta` move by `baseline x change_pct / 100`. |

Each prompt's question sentence is unique (for example "Estimate the percentage change in total factor productivity"),
because the mock LLM in `experiments/<model>/mock.yaml` answers by matching it.

Changing `persona` to a wealthy household changes the LLM's estimate only. The Krusell-Smith model still applies one `d`
path to every household; targeting a subgroup needs a household-model extension, not a different prompt.

## Rules

- Start with `{{ persona }}`, then `{% if narrative %}{{ narrative }}\n{% endif %}{{ event }}`, then the optional
  `model_info` block, then one question sentence, then `{% include "_answer_format.j2" %}` (copy an existing template).
- Do not state the variable's current level or ask for percentage points; units stay relative percent.
- Keep the question sentence free of the other prompts' wording.
- A new prompt type: add `<type>.j2` and an entry in `prompts.yaml`; add its name to an experiment's `prompts:` list.

## Tests

`tests/llms/test_prompts.py` renders every template with persona, narrative, event and model info, checks that each asks for
`change_pct`, and that every `prompts.yaml` entry has a template, a shockable variable and valid bounds.
