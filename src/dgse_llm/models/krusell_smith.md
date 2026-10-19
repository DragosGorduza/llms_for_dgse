# Model `krusell_smith` (heterogeneous households)

## The economy in plain words

Many households (a continuum) differ in two ways: their **income this period** (a random productivity or employment
draw that changes over time and is persistent) and **how much they have saved**. Households cannot borrow: assets
must stay above a limit (`amin`, 0 by default). So they save as a buffer against bad income draws ("precautionary
saving"). All savings are invested in **capital**, which one representative firm combines with labour to produce
output. Labour supply is fixed at 1 (people do not choose hours).

Why this matters: because households differ, how aggregate consumption reacts to a change in wages, interest rates or
taste depends on how many households have little savings and are close to the borrowing limit. Those households
spend almost every extra unit of income ("hand to mouth"). This is what the representative-household RBC model
misses, and it is the model in the SSJ notebook that the LLM-generated pandemic narrative drives.

Everyone knows the future path of the inputs from period 0 (perfect foresight). In period 0 the distribution of
households over income and assets is its steady-state shape.

**Time convention.** One period is a quarter by convention (`delta = 0.025`), not enforced. Rates are per period:
0.01 = 1%. "Rises" and "falls" always mean relative to the steady-state value.

**Household preferences.** In period t utility is `c^(1-1/eis)/(1-1/eis) - d_t * c`. The first-order condition is
`c_t^(-1/eis) - d_t = beta_t * E[(1+r_{t+1}) (c_{t+1}^(-1/eis) - d_{t+1})]`. This is the consistent form of the `d`
wedge in the notebook; the notebook's backward step uses `+ d` in the envelope condition and `dplus` in the
consumption rule, which do not correspond to one utility function.

## Variables, one by one

10 variables: 3 **inputs** (exogenous; an experiment, including an LLM-generated shock, can change them), 1
**unknown** (the solver finds it) and 6 **derived** (computed afterwards). `eis` is a constant parameter here, not a
shockable path.

### `Z` : productivity (TFP). Input, shockable.
- **Units:** index, about 0.88 in the steady state (calibrated so that output is 1). Allowed range: 0.05 to 5.
- **What it is:** how much output the firm gets from a given amount of capital and labour.
- **Macro interpretation:** the technology / efficiency shock.
- **If it rises:** a productivity boom: output, wages and the return on capital rise; households earn more and save more.
- **If it falls:** a productivity slump: output, wages and the return on capital fall; households earn less and save less.

### `d` : taste shifter (desire to postpone spending). Input, shockable.
- **Units:** marginal-utility units, 0 in the steady state. The marginal utility of consumption is about 1, so
  `d = 0.02` is roughly a 2% change in the value of spending today. Allowed range: -0.2 to 0.2.
- **What it is:** a term subtracted from the satisfaction of consuming today (marginal utility is `c^(-1/eis) - d`).
  It changes how much households WANT to spend without changing prices or incomes.
- **Macro interpretation:** non-economic behavioural change: fear, lockdowns, voluntary distancing, caution, or the
  opposite (pent-up demand, optimism). This is the input the notebook's pandemic narrative moves, and the main
  place where an LLM's estimate of behaviour (for example a % fall in social consumption) enters the model.
- **If it rises:** households want to spend less today. Consumption falls on impact and savings rise, so capital
  rises and the interest rate falls.
- **If it falls:** households want to spend more today (pent-up demand, confidence). Consumption rises on impact and
  savings fall, so capital falls and the interest rate rises.

### `beta` : the discount factor (patience). Input, shockable.
- **Units:** unitless, about 0.98 in the steady state (calibrated so that the interest rate is 1%). Allowed range: 0.5 to 1.0.
- **What it is:** how much households value period t+1 relative to period t. Closer to 1 = more patient.
- **Macro interpretation:** a saving mood; shifts spending between today and tomorrow.
- **If it rises:** households become more patient: they save more and consume less today, so capital builds up.
- **If it falls:** households become less patient: they save less and consume more today, so capital falls.

### `K` : capital stock. Unknown.
- **Units:** quantity in units of output, measured at the END of period t.
- **What it is:** the economy's machines and buildings. In equilibrium it equals total household savings (`A`).
  The firm uses it in period t+1.
- **Macro interpretation:** productive capacity; it moves slowly.
- **If it rises:** more capacity next period: higher output and wages, lower return on capital.
- **If it falls:** less capacity next period: lower output and wages, higher return on capital.

### `Y` : output (GDP). Derived.
- **Units:** quantity in units of output; 1 in the steady state.
- **What it is:** `Z * K_{t-1}^alpha` (labour is fixed at 1).
- **Macro interpretation:** the size of the economy.
- **If it rises:** a boom. **If it falls:** a recession.

### `r` : net return on capital (real interest rate). Derived.
- **Units:** rate per period; 0.01 = 1% in the steady state.
- **What it is:** `alpha Z K_{t-1}^(alpha-1) - delta`. High when capital is scarce or productive.
- **Macro interpretation:** the reward for saving.
- **If it rises:** saving is more rewarding: households save more and consume less.
- **If it falls:** saving is less rewarding: households save less and consume more.

### `w` : real wage. Derived.
- **Units:** units of output per unit of labour; about 0.89 in the steady state.
- **What it is:** `(1-alpha) Z K_{t-1}^alpha`. A household's labour income is `w` times its own income level.
- **Macro interpretation:** the reward for labour; the main source of income for most households.
- **If it rises:** households earn more, and those with little savings spend almost all of it: aggregate consumption rises.
- **If it falls:** households earn less: aggregate consumption falls.

### `A` : aggregate household assets. Derived.
- **Units:** quantity in units of output.
- **What it is:** the sum of all households' savings chosen at the end of period t. The model requires `A = K`.
- **Macro interpretation:** total saving of the economy.
- **If it rises:** households are saving more in total, pushing capital up.
- **If it falls:** households are saving less in total, pushing capital down.

### `C` : aggregate consumption. Derived.
- **Units:** quantity in units of output.
- **What it is:** the sum of what all households spend in period t.
- **Macro interpretation:** household demand, the main channel through which `d` shocks show up.
- **If it rises:** households spend more in total. **If it falls:** households spend less in total.

### `goods_mkt` : goods-market check. Derived.
- **Units:** quantity in units of output; 0 in the steady state.
- **What it is:** `Y - C - (K_t - (1-delta) K_{t-1})`: output minus consumption minus investment.
- **Macro interpretation:** not an economic variable but a consistency test (Walras' law). When the asset market
  clears this must be 0 in every period.
- **If it rises:** positive means a numerical error: output exceeds consumption plus investment.
- **If it falls:** negative means a numerical error: consumption plus investment exceed output.

## The equation the solver enforces (target)

### `asset_mkt` : the asset market clears
- **Equation:** `A_t - K_t = 0`.
- **In words:** households' total savings must equal the capital the firm uses. `A_t` comes from solving every
  household's problem backwards in time (given the paths of interest rate, wage, `beta`, `d`) and then moving the
  population forwards in time. If `A_t > K_t`, households want to save more than there is capital, so the interest
  rate must fall; the solver adjusts `K`.

## Parameters and defaults

| Parameter | Default | Meaning in words |
| --- | --- | --- |
| `alpha` | 0.11 | Share of output paid to capital (low, so households hold little capital income and more are constrained). |
| `delta` | 0.025 | Fraction of capital that wears out per period. |
| `eis` | 1.0 | Elasticity of intertemporal substitution: willingness to move spending across time (1 = log utility). |
| `rho_e` | 0.966 | Persistence of income shocks; closer to 1 = longer-lasting. |
| `sd_e` | 0.5 | Cross-sectional standard deviation of log income: income inequality. |
| `nE` | 7 | Number of income levels on the grid. |
| `nA` | 100 | Number of asset levels on the grid (small so tests are fast; use `nA: 500` to match the notebook). |
| `amin` | 0 | Borrowing limit (lowest assets); 0 = no borrowing. |
| `amax` | 200 | Highest asset level on the grid. |
| `agrid_curv` | 1000 | How strongly grid points are packed near the borrowing limit. |
| `target_r` | 0.01 | Steady-state return that `beta` is calibrated to hit. |
| `target_Y` | 1.0 | Steady-state output that fixes the level of `Z`. |
| `ss_tol` | 1e-11 | Convergence tolerance of the steady-state iterations. |
| `beta_lo`, `beta_hi` | 0.93, 0.986 | Search bracket for the calibrated discount factor. |

## Steady state

Labour is fixed at 1, so `K = alpha Y / (r + delta)`, `Z = Y / K^alpha`, `w = (1-alpha) Y`. The discount factor
`beta` is found with Brent's method so that household assets equal `K`.

## Timing

`r_t` and `w_t` depend on `K_{t-1}` and `Z_t`. `K_{-1}` is the steady-state value. After the last period the
household value function is at its steady state.

## Function reference (inputs and outputs)

All "paths" are `numpy` arrays of dtype float64 and shape `(T,)`. Household objects have shape `(nE, nA)`: one number
per (income level, asset level).

| Function | Inputs | Output |
| --- | --- | --- |
| `compute_steady_state()` | none | `dict[str, float]` with a scalar for each of the 10 variables. Also stores the household policies and distribution in `self._hh`. |
| `residuals(U, Z)` | `U` with key `K`; `Z` with keys `Z`, `d`, `beta` | `dict[str, ndarray]` with key `asset_mkt`, shape `(T,)`; 0 = market clears. |
| `outputs(U, Z)` | same as `residuals` | `dict[str, ndarray]` with all 10 variables, each `(T,)`. |
| `_simulate(U, Z)` | same as `residuals` | `dict[str, ndarray]` with `K_lag`, `r`, `w`, `A`, `C`, `Y`, each `(T,)`. |

The household helpers in `_het_agents.py` document their `(nE, nA)` inputs and outputs in their docstrings.

## Classic shocks (ready-made, no LLM)

These are defined in `classic_shocks` in `krusell_smith.py`. Use one in a config with `source: classic` and
`classic: <name>`. `intensity` multiplies the size (2 = twice as big, -1 = the opposite shock). Timing: **one-time** =
one period, then the model's own dynamics take over; **multi-period** = the same size for a number of periods;
**decaying** = starts at the size and shrinks by a factor `rho` each period. Sizes are changes in the input, added to
its baseline.

| Shock | Input | Size | Timing | Story |
| --- | --- | --- | --- | --- |
| `tfp_boom_persistent` | `Z` | +0.01 | decaying, rho 0.8 | A 1% productivity boom that fades quickly. |
| `tfp_slump_one_time` | `Z` | -0.02 | one-time | A one-period 2% productivity drop. |
| `precautionary_saving` | `d` | +0.02 | 6 periods | Households want to spend less (fear, caution) for 6 periods. |
| `patience_burst` | `beta` | +0.002 | one-time | Households save more and consume less for one period. |

### Classic shock `tfp_boom_persistent`
Output, wages and consumption rise. Households save part of the gain, so capital rises with a lag.

### Classic shock `tfp_slump_one_time`
Wages and output fall for one period. Households with little savings cut consumption the most, so aggregate
consumption reacts more than in a representative-household model.

### Classic shock `precautionary_saving`
The classic counterpart of the LLM pandemic shock: the same input `d`, with a fixed size and timing instead of values
from an LLM. Consumption falls and saving rises while it lasts.

### Classic shock `patience_burst`
A single period of extra patience: consumption dips, assets and capital rise.

## What to expect

Steady state: `r = 0.01`, `Y = 1`, `A = K`. A positive `Z` shock raises `Y`, `w`, `C`, and `K` rises with a lag. A
positive `d` shock lowers `C` on impact and raises `K` afterwards. `goods_mkt` is about 0 on every path.
