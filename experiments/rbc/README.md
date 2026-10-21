# experiments/rbc

Representative household, capital and elastic labour. Plain-language guide:
[src/dgse_llm/models/rbc.md](../../src/dgse_llm/models/rbc.md). Machine-readable register of every parameter and
variable: [model.json](model.json) (checked against the code by `tests/experiments/test_register.py`).

## What can be shocked

| Variable | Shockable | Prompt (`prompts/rbc/`) | The LLM answers |
| --- | --- | --- | --- |
| `Z` TFP | yes | `tfp` | % change in TFP |
| `beta` discount factor | yes | `discount_factor` | % change in the discount factor |
| `K` capital, `L` hours | no, unknowns: the solver finds them | | |
| `Y`, `C`, `I`, `r`, `w` | no, derived: reported only | | |

Parameters (`alpha`, `delta`, `beta`, `sigma`, `phi`) are never shocked; change them in the experiment's `model:` entry
(`model: {name: rbc, params: {alpha: 0.4}}`).
Note that the steady-state `beta` parameter and the shockable `beta` path are different things.

## Experiments here

| File | Contents |
| --- | --- |
| `generic.yaml` | Both prompts x all personas x 6 scenarios x model info on/off, on two OpenAI settings (temperature 0 and 1, 5 repeats at 1), T = 120 |
| `mock.yaml` | Small offline version with a mock LLM, T = 30 |

Run: `dgse-llm run experiments/rbc/mock.yaml --report`.
