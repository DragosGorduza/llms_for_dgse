# experiments/nk3

Three-equation New Keynesian model (IS curve, Phillips curve, Taylor rule). Every variable is a deviation from a
steady state, so 0 is normal. Plain-language guide: [src/dgse_llm/models/nk3.md](../../src/dgse_llm/models/nk3.md).
Machine-readable register of every parameter and variable: [model.json](model.json) (checked against the code by
`tests/experiments/test_register.py`).

## What can be shocked

| Variable | Shockable | Prompt (`prompts/nk3/`) | The LLM answers | Units per +1% (see `prompts.yaml`) |
| --- | --- | --- | --- | --- |
| `r_n` natural rate (demand) | yes | `demand` | % change in aggregate demand | 0.01 |
| `u` cost push | yes | `cost_push` | % change in firms' production costs | 0.0002 |
| `eps_i` policy shock | yes | `monetary_policy` | % change in the policy rate vs the rule | 0.0001 |
| `wedge_pi` expectations wedge | yes | `inflation_expectations` | % change in expected inflation | 0.00005 |
| `x` output gap, `pi` inflation, `i` policy rate | no, unknowns: the solver finds them | | | |
| `rr` real rate | no, derived: reported only | | | |

All four inputs have baseline 0, so a relative change needs a reference level; the conversions in the last column are
calibrations written down in `prompts/nk3/prompts.yaml`. Parameters (`beta`, `sigma`, `kappa`, `phi_pi`, `phi_x`) are
never shocked; change them in the experiment's `model:` entry (`model: {name: nk3, params: {kappa: 0.2}}`).

## Experiments here

| File | Contents |
| --- | --- |
| `generic.yaml` | All 4 prompts x all personas x 6 scenarios x model info on/off, on two OpenAI settings (temperature 0 and 1, 5 repeats at 1), T = 120 |
| `mock.yaml` | Small offline version with a mock LLM, T = 30 |

Run: `dgse-llm run experiments/nk3/mock.yaml --report`.
