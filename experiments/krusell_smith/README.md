# experiments/krusell_smith

Heterogeneous households that cannot borrow, one asset (capital), one firm, fixed labour. Plain-language guide:
[src/dgse_llm/models/krusell_smith.md](../../src/dgse_llm/models/krusell_smith.md). Machine-readable register of
every parameter and variable: [model.json](model.json) (checked against the code by `tests/experiments/test_register.py`).

## What can be shocked

| Variable | Shockable | Prompt (`prompts/krusell_smith/`) | The LLM answers |
| --- | --- | --- | --- |
| `Z` TFP | yes | `tfp` | % change in TFP |
| `beta` discount factor | yes | `discount_factor` | % change in the discount factor |
| `d` taste shifter | yes | `household_consumption` | % change in desired consumption (converted to `d`: -40% gives `d` = +0.04) |
| `K` capital | no, unknown: the solver finds it | | |
| `Y`, `r`, `w`, `A`, `C`, `goods_mkt` | no, derived: reported only | | |

Parameters are never shocked. `alpha`, `delta`, `eis`, `rho_e`, `sd_e`, `amin` are economic; `nE`, `nA`, `amax`,
`agrid_curv`, `target_r`, `target_Y`, `ss_tol`, `beta_lo`, `beta_hi` are numerical or calibration settings. Change them
in the experiment's `model:` entry. With `include_model_info` on, the prompt names the model and lists them.

## Experiments here

| File | Contents |
| --- | --- |
| `generic.yaml` | All 3 prompts x all personas x 6 scenarios x model info on/off, on two OpenAI settings (temperature 0 and 1, 5 repeats at 1), T = 120 |
| `mock.yaml` | Small offline version with a mock LLM, T = 30 |

Solver and parameter variants (for example the notebook's grid) go in the experiment's `model:` entry:
`model: {name: krusell_smith, params: {nA: 500}}`.

Run: `dgse-llm run experiments/krusell_smith/mock.yaml --report`. The grid is slow here (each cell solves a
heterogeneous-agent model); use a small `T` while developing.
