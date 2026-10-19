# Model `rbc` (real business cycle)

## The economy in plain words

One representative household owns all the capital (machines, buildings) and also supplies all the labour. Each
period it rents its capital and its hours to a firm. The firm uses them to produce output. The household then
decides three things: how much of its income to **consume** now, how much to **save** (which becomes new capital
next period), and how many **hours to work**. Output that is not consumed is invested.

Only two things from outside move this economy: **productivity** (`Z`) and **patience** (`beta`). Everything else
reacts to them. Everyone knows the whole future path of `Z` and `beta` from period 0 ("perfect foresight"), so
households react in advance to news about the future.

What the model is good for: showing how productivity or taste changes move output, consumption, investment, hours,
wages and interest rates together. What it cannot do: it has no unemployment, no inflation, no money, and no
differences between households.

**Time convention.** One period is a quarter by convention (the depreciation rate 0.025 is a quarterly rate), but
the code does not enforce it. All rates are per period: 0.01 = 1% per period.

**Deviations.** The steady state is the long-run resting point. A variable "rises" or "falls" below always means
relative to its steady-state value, not relative to the previous period.

## Variables, one by one

There are 9 variables: 2 **inputs** (exogenous; an experiment, including an LLM-generated shock, can change them),
2 **unknowns** (the solver finds them) and 5 **derived** (computed from the others, only reported).

### `Z` : productivity (TFP). Input, shockable.
- **Units:** index, equal to 1 in the steady state. Allowed range for experiments: 0.05 to 5.
- **What it is:** Total factor productivity. How much output the economy gets out of a given amount of capital and
  labour. Think of it as the state of technology and general efficiency.
- **Macro interpretation:** the standard source of business cycles in an RBC model. It is not explained by the model.
- **If it rises:** a productivity boom (new technology, efficiency gains). The same inputs produce more, so output,
  wages and the return on capital rise, and the household works and invests more.
- **If it falls:** a productivity slump (supply disruption, loss of efficiency). Output, wages and the return on
  capital fall; the household works and invests less.

### `beta` : the discount factor (patience). Input, shockable.
- **Units:** unitless, about 0.99 in the steady state. Allowed range: 0.5 to 1.2. In period t it measures how much
  the household values utility in period t+1 relative to utility in period t.
- **What it is:** a taste parameter. Close to 1 means very patient; lower means the household prefers the present.
- **Macro interpretation:** a "saving mood" or demand shock. Changes in `beta` shift spending between today and
  tomorrow without any change in technology.
- **If it rises:** the household becomes more patient or more cautious about the future. It consumes less now and
  saves and invests more; capital builds up and future consumption is higher.
- **If it falls:** the household becomes impatient (a "live for today" mood). It consumes more now and invests less;
  capital falls and future consumption is lower.

### `K` : capital stock. Unknown.
- **Units:** quantity, in units of output, measured at the END of period t.
- **What it is:** the machines, buildings and equipment the household owns. It is chosen in period t and used for
  production in period t+1.
- **Macro interpretation:** the economy's productive capacity. It moves slowly, which is why shocks to `Z` and
  `beta` have long-lasting effects.
- **If it rises:** more capacity next period: higher output and wages, but a lower return on each unit of capital.
- **If it falls:** less capacity next period: lower output and wages, but a higher return on each unit.

### `L` : hours worked (labour). Unknown.
- **Units:** quantity of hours, equal to 1 in the steady state (a normalisation).
- **What it is:** how much the household works in period t. It chooses `L` by comparing the wage with the value
  of leisure.
- **Macro interpretation:** the employment margin of the economy.
- **If it rises:** people work more, so output is higher today. This happens when wages are high, or when the
  household feels poorer and wants to earn more.
- **If it falls:** people work less, so output is lower today. This happens when wages are low, or when the household
  feels richer.

### `Y` : output (GDP). Derived.
- **Units:** quantity, in units of output.
- **What it is:** everything produced in period t: `Z * K_{t-1}^alpha * L^(1-alpha)`. It is also the total income
  of the economy.
- **Macro interpretation:** the size of the economy, the headline measure of the cycle.
- **If it rises:** a boom; more production and income.
- **If it falls:** a recession; less production and income.

### `C` : consumption. Derived.
- **Units:** quantity, in units of output.
- **What it is:** what the household uses up in period t: `Y + (1-delta) K_{t-1} - K_t`.
- **Macro interpretation:** household spending, the largest component of demand. It is smoother than output because
  the household saves in good times and dissaves in bad times.
- **If it rises:** the household is better off, more impatient, or saving less.
- **If it falls:** the household is worse off, more patient, or saving more.

### `I` : investment. Derived.
- **Units:** quantity, in units of output.
- **What it is:** output turned into new capital: `K_t - (1-delta) K_{t-1}`. In the steady state it just replaces
  depreciation: `delta * K`.
- **Macro interpretation:** the most volatile part of demand; it responds strongly to expected future returns.
- **If it rises:** the capital stock is growing (good news about the future, or more patience).
- **If it falls:** the capital stock is growing more slowly or shrinking (bad news about the future, or impatience).

### `r` : net return on capital (real interest rate). Derived.
- **Units:** rate per period; 0.01 = 1%.
- **What it is:** the rent paid per unit of capital in period t minus depreciation:
  `alpha Z (K_{t-1}/L)^(alpha-1) - delta`.
- **Macro interpretation:** the reward for saving. It is what guides the consumption-saving choice.
- **If it rises:** saving is more rewarding (capital is scarce or more productive): households shift consumption into
  the future.
- **If it falls:** saving is less rewarding (capital is plentiful or less productive): households shift consumption
  towards the present.

### `w` : real wage. Derived.
- **Units:** units of output per hour.
- **What it is:** what the firm pays for one hour of labour: `(1-alpha) Z (K_{t-1}/L)^alpha`.
- **Macro interpretation:** the reward for working; the household's main source of income growth.
- **If it rises:** work pays more: the household supplies more hours and earns more.
- **If it falls:** work pays less: the household supplies fewer hours and earns less.

## The two equations the solver enforces (targets)

The solver searches for paths of `K` and `L` that make both of these zero in every period. Each is an optimality
condition of the household.

### `euler` : the consumption-saving condition
- **Equation:** `C_t^-sigma = beta_t * C_{t+1}^-sigma * (1 + r_{t+1})`.
- **In words:** giving up one unit of consumption today and consuming the proceeds tomorrow must leave the household
  exactly as well off. If today's side is bigger, the household would rather consume more now and save less.
- **Why it matters:** it ties today's saving to the future return `r` and to patience `beta`. A change in `beta` or in
  expected future productivity moves `K` through this equation.

### `labor` : the work-versus-leisure condition
- **Equation:** `chi * L_t^phi = w_t * C_t^-sigma`.
- **In words:** the pain of working one more hour must equal the extra consumption the wage buys. If the left side is
  bigger, the household would rather work less.
- **Why it matters:** it makes hours respond to the wage and to wealth. `chi` is chosen so that steady-state hours are 1.

The other relationships (production, resource constraint, prices) are identities computed inside `_core` in `rbc.py`.

## Parameters and defaults

| Parameter | Default | Meaning in words | Raising it makes... |
| --- | --- | --- | --- |
| `alpha` | 0.33 | Share of output paid to capital. | capital more important; output more sensitive to `K`. |
| `delta` | 0.025 | Fraction of capital that wears out per period. | capital depreciate faster; more investment is needed just to stand still. |
| `beta` | 0.99 | Steady-state patience. Sets the steady-state return `r = 1/beta - 1`. | households more patient; larger capital stock and lower steady-state return. |
| `sigma` | 1.0 | How much the household dislikes uneven consumption over time. | consumption smoother over time; weaker response to interest rates. |
| `phi` | 1.0 | How reluctant the household is to change hours (inverse Frisch elasticity). | hours respond less to wages. |

## Steady state and timing

`r = 1/beta - 1`, `L = 1`, `K/L = (alpha/(r+delta))^(1/(1-alpha))`. Before period 0 the economy sits at the steady
state (`K_{-1}` equals the steady-state capital); after the last period it is assumed to be back at the steady state.

## Function reference (inputs and outputs)

All "paths" are `numpy` arrays of dtype float64 and shape `(T,)`, one number per period. `U` is a dict of the unknown
paths, `Z` a dict of the input paths.

| Function | Inputs | Output |
| --- | --- | --- |
| `compute_steady_state()` | none (uses the parameters) | `dict[str, float]` with a scalar for each of `Z`, `beta`, `K`, `L`, `Y`, `C`, `I`, `r`, `w`. Also stores the labour weight `chi`. |
| `residuals(U, Z)` | `U` with keys `K`, `L`; `Z` with keys `Z`, `beta` | `dict[str, ndarray]` with keys `euler`, `labor`; each `(T,)`; 0 = condition holds. |
| `outputs(U, Z)` | same as `residuals` | `dict[str, ndarray]` with all nine variables, each `(T,)`. |
| `_core(U, Z)` | same as `residuals` | tuple of five `(T,)` arrays: capital at start of period, `Y`, `r`, `w`, `C`. |

The docstrings in `rbc.py` give the same information next to the code.

## Classic shocks (ready-made, no LLM)

These are defined in `classic_shocks` in `rbc.py`. Use one in a config with `source: classic` and `classic: <name>`.
`intensity` multiplies the size (2 = twice as big, -1 = the opposite shock). Timing: **one-time** = one period, then the
model's own dynamics take over; **multi-period** = the same size for a number of periods; **decaying** = starts at the
size and shrinks by a factor `rho` each period. Sizes are changes in the input, added to its baseline.

| Shock | Input | Size | Timing | Story |
| --- | --- | --- | --- | --- |
| `tfp_boom_persistent` | `Z` | +0.01 | decaying, rho 0.9 | A 1% productivity boom that fades slowly. |
| `tfp_slump_one_time` | `Z` | -0.02 | one-time | A one-period 2% productivity drop (short supply disruption). |
| `tfp_slump_multi_period` | `Z` | -0.01 | 4 periods | A year-long 1% slump, then an abrupt return to normal. |
| `impatience_wave` | `beta` | -0.005 | 4 periods | Households become less patient: consume more, invest less. |

### Classic shock `tfp_boom_persistent`
The textbook technology shock. Output, hours, consumption and investment rise together and then return to normal as
the boom fades. Capital adjusts slowly, so effects outlast the shock.

### Classic shock `tfp_slump_one_time`
One bad period. The economy falls on impact and recovers through capital and hours; households smooth consumption, so
consumption falls by less than output.

### Classic shock `tfp_slump_multi_period`
A year-long slump known from period 0. Households see the end coming and smooth consumption; the abrupt end causes a
rebound.

### Classic shock `impatience_wave`
For 4 periods households value the future less. Consumption jumps up, investment and capital fall, and output is lower
afterwards because there is less capital.

## What to expect

A temporary positive `Z` shock raises `Y`, `C`, `I` and `L`, then decays back to zero as capital returns to its
steady state (slowly). A fall in `beta` raises `C` and lowers `K` on impact. A zero shock gives a zero response.
