# Model `nk3` (three-equation New Keynesian)

## The economy in plain words

This is the small "workhorse" model central banks use to think about inflation and the business cycle. It has
three relationships:

1. **Demand (the IS curve).** Households spend more when the real interest rate is low, and expect to spend more when
   they expect future activity to be strong.
2. **Pricing (the Phillips curve).** Firms raise prices faster when demand is strong, when they expect inflation to be
   high, and when their costs are pushed up.
3. **Policy (the Taylor rule).** The central bank raises the interest rate when inflation or output are above
   normal, and cuts it when they are below.

The model is written in **deviations from a steady state with zero inflation**. Every variable is 0 when nothing
happens. Positive means "above normal", negative means "below normal". All values are decimals per period, so
0.01 means 1 percentage point. Everyone knows the future path of the inputs from period 0 (perfect foresight).

The special input `wedge_pi` lets households and firms hold a **biased view of future inflation**, without changing
any structural parameter. It is the natural slot for an LLM-generated or survey-based inflation belief.

What the model is good for: how demand, supply, monetary and expectations shocks move inflation, output and interest
rates. What it cannot do: it has no capital, no unemployment, no heterogeneity, and it is linear (the response to a
shock twice as big is exactly twice as big).

## Variables, one by one

8 variables: 4 **inputs** (exogenous; an experiment, including an LLM-generated shock, can change them), 3
**unknowns** (the solver finds them) and 1 **derived** (computed afterwards). "Rises" and "falls" mean relative to the
steady state, which is 0 for every variable.

### `r_n` : the natural rate of interest (demand). Input, shockable.
- **Units:** decimal per period, 0 in the steady state. Allowed range: -0.5 to 0.5.
- **What it is:** the real interest rate at which demand would exactly equal normal supply. It summarises everything
  that shifts spending independently of the interest rate: confidence, fiscal policy, credit conditions, the desire
  to save.
- **Macro interpretation:** the demand shifter.
- **If it rises:** demand is stronger at any given interest rate (more confidence, looser fiscal policy, easier
  credit). The output gap and inflation rise, and the central bank raises rates.
- **If it falls:** demand is weaker (confidence falls, households save more, credit tightens). The output gap and
  inflation fall, and the central bank cuts rates.

### `u` : cost-push shock. Input, shockable.
- **Units:** decimal per period, 0 in the steady state. Allowed range: -0.5 to 0.5.
- **What it is:** a supply-side force that raises firms' prices for a given level of demand: an energy price spike, a
  supply-chain disruption, a jump in wage demands.
- **Macro interpretation:** the supply shock. It creates the dilemma central banks fear: inflation and weak output at
  the same time.
- **If it rises:** inflation is pushed up although demand has not changed. The central bank tightens in response, so
  output falls (stagflation).
- **If it falls:** inflation is pushed down for a given level of demand (cheaper energy, easier supply). The central
  bank can ease and output rises.

### `eps_i` : monetary policy shock. Input, shockable.
- **Units:** decimal per period, 0 in the steady state. Allowed range: -0.5 to 0.5.
- **What it is:** a surprise deviation of the interest rate from what the central bank's rule would normally set,
  not explained by current inflation or output.
- **Macro interpretation:** the pure policy shock (a surprise hike or cut).
- **If it rises:** a surprise tightening. Borrowing costs more, so demand, output and inflation fall.
- **If it falls:** a surprise easing. Borrowing is cheaper, so demand, output and inflation rise.

### `wedge_pi` : inflation-expectations wedge. Input, shockable.
- **Units:** decimal per period, 0 in the steady state. Allowed range: -0.5 to 0.5.
- **What it is:** how much households' and firms' EXPECTED next-period inflation exceeds the model-consistent
  (rational) value. 0 means expectations are rational. It enters both the demand and the pricing equation.
- **Macro interpretation:** de-anchored or biased expectations (for example after a news story about prices).
- **If it rises:** people expect more inflation than the model implies. Firms raise prices today, so inflation rises.
  Households see a lower real interest rate and want to spend more. Output can still fall if the central bank
  over-reacts to the extra inflation (when `phi_pi * beta > 1`; see the closed form below).
- **If it falls:** people expect less inflation than the model implies. Firms raise prices less, so inflation falls,
  and households see a higher real interest rate. Output can rise if the central bank over-eases.

### `x` : the output gap. Unknown.
- **Units:** decimal, 0 in the steady state (0.01 = output 1% above its normal level).
- **What it is:** how far output is above or below its normal (potential) level.
- **Macro interpretation:** how hot or cold the economy is.
- **If it rises:** the economy is overheating: strong demand, tight labour market, upward pressure on inflation.
- **If it falls:** the economy is slack: weak demand, a recession, downward pressure on inflation.

### `pi` : inflation. Unknown.
- **Units:** decimal per period, 0 in the steady state (0.01 = 1 percentage point above the target per period).
- **What it is:** the rate of change of the price level, measured as a deviation from the central bank's target.
- **Macro interpretation:** the central bank's main objective variable.
- **If it rises:** prices rise faster than the target. Purchasing power erodes and the central bank is pushed to tighten.
- **If it falls:** inflation is below the target (disinflation). The central bank is pushed to ease.

### `i` : the nominal interest rate. Unknown.
- **Units:** decimal per period, 0 in the steady state.
- **What it is:** the policy interest rate set by the central bank, as a deviation from its normal level.
- **Macro interpretation:** the central bank's instrument.
- **If it rises:** tighter monetary policy. Borrowing is costlier, which cools demand and inflation.
- **If it falls:** looser monetary policy. Borrowing is cheaper, which stimulates demand and inflation.

### `rr` : the ex-post real interest rate. Derived.
- **Units:** decimal per period, 0 in the steady state.
- **What it is:** the nominal rate minus the inflation that actually happens next period: `i_t - pi_{t+1}`.
- **Macro interpretation:** the true cost of borrowing once inflation is accounted for.
- **If it rises:** saving pays more in real terms and borrowing costs more, so households spend less today.
- **If it falls:** saving pays less in real terms and borrowing is cheaper, so households spend more today.

## The three equations the solver enforces (targets)

The solver searches for paths of `x`, `pi` and `i` that make all three zero in every period. Write
`pi_e_t = pi_{t+1} + wedge_pi_t` for the inflation people EXPECT.

### `is_curve` : demand
- **Equation:** `x_t = x_{t+1} - (1/sigma) (i_t - pi_e_t - r_n_t)`.
- **In words:** today's output gap equals tomorrow's, minus the effect of the expected real interest rate
  `i_t - pi_e_t` being above the natural rate `r_n_t`. A high real rate makes people postpone spending.

### `phillips` : pricing
- **Equation:** `pi_t = beta pi_e_t + kappa x_t + u_t`.
- **In words:** inflation today is driven by expected inflation, by how hot the economy is (`x_t`), and by
  cost-push shocks (`u_t`).

### `taylor` : central bank rule
- **Equation:** `i_t = phi_pi pi_t + phi_x x_t + eps_i_t`.
- **In words:** the central bank raises the interest rate by `phi_pi` points for each point of inflation and by
  `phi_x` for each point of the output gap, plus any policy surprise `eps_i_t`.

## Parameters and defaults

| Parameter | Default | Meaning in words | Raising it makes... |
| --- | --- | --- | --- |
| `beta` | 0.99 | How much firms and households value next period relative to today. | prices more forward-looking; expected inflation matters more. |
| `sigma` | 1.0 | How little demand reacts to the real interest rate. | demand less sensitive to interest rates. |
| `kappa` | 0.1 | How strongly inflation reacts to the output gap (slope of the Phillips curve). | prices more flexible; inflation reacts more to a hot economy. |
| `phi_pi` | 1.5 | Central bank reaction to inflation. Must exceed 1 for a stable economy (the Taylor principle). | a more aggressive central bank; inflation shocks die out faster but output falls more. |
| `phi_x` | 0.125 | Central bank reaction to the output gap. | a central bank that stabilises output more. |

## Function reference (inputs and outputs)

All "paths" are `numpy` arrays of dtype float64 and shape `(T,)`, one number per period. `U` is a dict of the unknown
paths, `Z` a dict of the input paths.

| Function | Inputs | Output |
| --- | --- | --- |
| `compute_steady_state()` | none | `dict[str, float]`: 0.0 for each of the 8 variables. |
| `residuals(U, Z)` | `U` with keys `x`, `pi`, `i`; `Z` with keys `r_n`, `u`, `eps_i`, `wedge_pi` | `dict[str, ndarray]` with keys `is_curve`, `phillips`, `taylor`; each `(T,)`; 0 = equation holds. |
| `outputs(U, Z)` | same as `residuals` | `dict[str, ndarray]` with all 8 variables, each `(T,)`. |

The docstrings in `nk3.py` give the same information next to the code.

## Classic shocks (ready-made, no LLM)

These are defined in `classic_shocks` in `nk3.py`. Use one in a config with `source: classic` and `classic: <name>`.
`intensity` multiplies the size (2 = twice as big, -1 = the opposite shock). Timing: **one-time** = one period, then the
model's own dynamics take over; **multi-period** = the same size for a number of periods. Sizes are changes in the
input (decimals per period), added to its baseline of 0.

| Shock | Input | Size | Timing | Story |
| --- | --- | --- | --- | --- |
| `demand_slump` | `r_n` | -0.005 | 4 periods | Confidence falls; the natural rate drops by 0.5 percentage points for a year. |
| `cost_push_energy` | `u` | +0.002 | 4 periods | Energy price spike adds 0.2 percentage points to inflation for a year. |
| `monetary_tightening` | `eps_i` | +0.0025 | one-time | A surprise 0.25 percentage point rate hike. |
| `expectations_unanchoring` | `wedge_pi` | +0.003 | 8 periods | People expect 0.3 percentage points more inflation than the model implies. |

### Classic shock `demand_slump`
Weaker demand: the output gap and inflation fall and the central bank cuts rates. It is the mirror image of a boom.

### Classic shock `cost_push_energy`
Stagflation: inflation rises while output falls, because the central bank tightens in response to inflation.

### Classic shock `monetary_tightening`
A pure policy surprise lasting one period. Output and inflation fall on impact and are back to normal the next period.

### Classic shock `expectations_unanchoring`
Biased inflation expectations. Inflation rises; with the default policy rule (`phi_pi * beta > 1`) output falls on
impact, as in the closed form below. This is the classic counterpart of the LLM-generated wedge (`nk_wedge_energy`).

## What to expect

The model is linear, so one Newton step is exact. For a one-period shock the economy is back at the steady state
from t=1, and the impact response has a closed form (with `k = 1 + phi_pi*kappa + phi_x` and `sigma = 1`):

| Shock at t=0 | `x_0` | `pi_0` |
| --- | --- | --- |
| `eps_i = e` | `-e / k` | `kappa x_0` |
| `u = u` | `-phi_pi u / k` | `u + kappa x_0` |
| `r_n = r` | `r / k` | `kappa x_0` |
| `wedge_pi = w` | `w (1 - phi_pi beta) / k` | `beta w + kappa x_0` |

Note the last row: with `phi_pi * beta > 1` the policy rule over-reacts to the inflation wedge, so the real rate rises
and output falls on impact even though inflation rises. After the horizon all variables are 0.
