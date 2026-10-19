# prompts/

Jinja2 templates rendered by `dgse_llm.llms.PromptRenderer`. Referenced by name from a shock's `llm.template`.

Variables available to every template: `narrative` (the current event text, empty if the shock has no events),
`t` (period index, only for narrative timelines) and everything under the shock's `llm.template_vars`
(for example `persona`). Undefined variables raise an error.

| Template | Asks for | Output keys |
| --- | --- | --- |
| `household_consumption.j2` | Change in social and remote consumption vs the pre-event level (from the KS notebook) | `social_change_pct`, `remote_change_pct` |
| `inflation_expectations.j2` | Change in expected inflation caused by the narrative | `inflation_expectation_change_pp` |

## How a template is connected

A template is used by a shock config (`configs/shocks/<name>.yaml`, field `llm.template`). The JSON keys the template asks
for must equal `llm.keys` there (`tests/test_configs.py` checks this). The key named by `llm.key` is converted by
`llm.mapping` into the units of the model input given by `variable`, so the prompt's units (percent, percentage points)
and the mapping's slope must agree. See the wiring table in the root README and `src/dgse_llm/shocks/README.md`.

## Rules

- End the prompt by asking for one JSON object with the exact keys listed in the shock's `llm.keys`.
- Ask for plain numbers in stated units; the answer is validated with pydantic (finite float per key, optional `llm.answer_bounds` range), so set `answer_bounds` in the shock config to the plausible range of those units.
- State units (percent, percentage points) and sign convention in the prompt; the mapping function in the shock config converts them to model units.
- Keep the instruction text and the persona constant across periods; only `narrative` should change, so differences in answers come from the narrative.

## Tests

`tests/llms/test_prompts.py` renders every template here with sample variables, and fails on any template that needs an undeclared variable.
