"""Three-equation New Keynesian (NK) model with an inflation-expectations wedge. Plain-language guide: nk3.md.

The economy in one paragraph: this is the small 'workhorse' model central banks use to think about inflation and the
business cycle. Three relationships describe it: (1) demand: households spend more when the real interest rate is low
(the IS curve); (2) supply: firms raise prices faster when demand is strong and when they expect high inflation (the
Phillips curve); (3) policy: the central bank raises the interest rate when inflation or output are high (the Taylor
rule). The model is written in DEVIATIONS from a steady state with zero inflation, so every variable is 0 when nothing
happens, positive means 'above normal' and negative means 'below normal'. All values are decimals per period
(0.01 = 1 percentage point). Everyone knows the future path of the inputs from period 0 (perfect foresight), except
that the `wedge_pi` input lets households and firms hold a biased view of future inflation.
"""

from __future__ import annotations

import numpy as np

from .base import ClassicShock, Model, Variable, lead, register_model


@register_model("nk3")
class NK3(Model):
    """Three-equation NK model. Unknowns: output gap `x`, inflation `pi`, interest rate `i`.

    Inputs (shockable): `r_n` demand, `u` cost-push, `eps_i` monetary policy, `wedge_pi` expectations bias.
    Parameters (`default_params`, all float):
      beta    discount factor (0.99): how much firms and households value next period relative to today.
      sigma   inverse intertemporal elasticity of substitution (1.0): higher = demand reacts less to the interest rate.
      kappa   slope of the Phillips curve (0.1): how strongly inflation rises when output is above normal.
      phi_pi  central bank reaction to inflation (1.5): points of interest rate per point of inflation. Must exceed 1
              for a stable economy (the 'Taylor principle').
      phi_x   central bank reaction to the output gap (0.125).
    """

    name = "nk3"
    variables = (
        Variable(
            "r_n", "exogenous", "decimal per period, 0 in the steady state",
            "Natural rate of interest: the real interest rate at which demand would exactly equal normal supply. It "
            "summarises everything that shifts spending independently of the interest rate: confidence, fiscal policy, "
            "credit conditions, saving behaviour. Also called the demand shifter.",
            shockable=True, bounds=(-0.5, 0.5),
            if_up="Stronger demand (households want to spend more at any given interest rate): the output gap and "
                  "inflation rise, and the central bank raises interest rates.",
            if_down="Weaker demand (confidence falls, households save more, credit tightens): the output gap and "
                    "inflation fall, and the central bank cuts rates.",
        ),
        Variable(
            "u", "exogenous", "decimal per period, 0 in the steady state",
            "Cost-push shock: a supply-side force that raises firms' prices for a given level of demand, such as an "
            "energy price spike, supply-chain disruption or a rise in wage demands.",
            shockable=True, bounds=(-0.5, 0.5),
            if_up="Inflation is pushed up even though demand has not changed; the central bank tightens, so output falls "
                  "(stagflation).",
            if_down="Inflation is pushed down for a given level of demand (cheaper energy, easier supply); the central "
                    "bank can ease and output rises.",
        ),
        Variable(
            "eps_i", "exogenous", "decimal per period, 0 in the steady state",
            "Monetary policy shock: a surprise deviation of the central bank's interest rate from what its rule says, "
            "for example a surprise hike or cut not explained by current inflation or output.",
            shockable=True, bounds=(-0.5, 0.5),
            if_up="A surprise tightening: borrowing becomes more expensive, so demand, output and inflation fall.",
            if_down="A surprise easing: borrowing becomes cheaper, so demand, output and inflation rise.",
        ),
        Variable(
            "wedge_pi", "exogenous", "decimal per period, 0 in the steady state",
            "Inflation-expectations wedge: how much households and firms' EXPECTED next-period inflation exceeds the "
            "model-consistent (rational) value. 0 means expectations are rational. This is the slot where an "
            "LLM-generated or survey-based belief can enter without changing any structural parameter.",
            shockable=True, bounds=(-0.5, 0.5),
            if_up="People expect more inflation than the model implies: firms raise prices today (inflation up), and "
                  "households see a lower real interest rate and want to spend more. Output can still fall if the "
                  "central bank over-reacts to the extra inflation (phi_pi * beta > 1).",
            if_down="People expect less inflation than the model implies: firms raise prices less (inflation down) and "
                    "households see a higher real interest rate. Output can rise if the central bank over-eases.",
        ),
        Variable(
            "x", "unknown", "decimal, 0 in the steady state (0.01 = output 1% above normal)",
            "Output gap: how far output is above or below its 'normal' (potential) level. Measures how hot or cold the "
            "economy is.",
            if_up="The economy is overheating: strong demand, tight labour market, upward pressure on inflation.",
            if_down="The economy is slack: a recession or weak demand, rising unemployment, downward pressure on inflation.",
        ),
        Variable(
            "pi", "unknown", "decimal per period, 0 in the steady state",
            "Inflation: the rate of change of the price level, as a deviation from the central bank's target (the steady "
            "state has inflation on target).",
            if_up="Prices rise faster than the target: purchasing power erodes and the central bank is pushed to tighten.",
            if_down="Inflation is below the target (disinflation): prices rise more slowly than the central bank wants "
                    "and it is pushed to ease.",
        ),
        Variable(
            "i", "unknown", "decimal per period, 0 in the steady state",
            "Nominal interest rate set by the central bank (the policy rate), as a deviation from its steady-state "
            "level.",
            if_up="Tighter monetary policy: borrowing is costlier, which cools demand and inflation.",
            if_down="Looser monetary policy: borrowing is cheaper, which stimulates demand and inflation.",
        ),
        Variable(
            "rr", "derived", "decimal per period, 0 in the steady state",
            "Ex-post real interest rate: the nominal rate minus next period's actual inflation, i(t) - pi(t+1). It is "
            "the true cost of borrowing once inflation is accounted for.",
            if_up="Saving pays more in real terms and borrowing costs more: households spend less today.",
            if_down="Saving pays less in real terms and borrowing is cheaper: households spend more today.",
        ),
    )
    # is_curve: demand equation. phillips: pricing equation. taylor: central bank rule.
    targets = ("is_curve", "phillips", "taylor")
    # Ready-made shocks that need no LLM. Reference them in a config with `source: classic, classic: <name>`.
    classic_shocks = (
        ClassicShock(
            "demand_slump", "r_n", -0.005, "multi_period", duration=4,
            description="A fall in confidence lowers the natural rate by 0.5 percentage points for 4 periods. Output "
                        "and inflation fall and the central bank cuts rates.",
        ),
        ClassicShock(
            "cost_push_energy", "u", 0.002, "multi_period", duration=4,
            description="An energy price spike adds 0.2 percentage points to inflation for 4 periods. Inflation rises and "
                        "output falls (stagflation) because the central bank tightens.",
        ),
        ClassicShock(
            "monetary_tightening", "eps_i", 0.0025, "one_time",
            description="A one-period surprise rate hike of 0.25 percentage points. Output and inflation fall on impact "
                        "and are back to normal the next period.",
        ),
        ClassicShock(
            "expectations_unanchoring", "wedge_pi", 0.003, "multi_period", duration=8,
            description="Households and firms expect 0.3 percentage points more inflation than the model implies for 8 "
                        "periods. Inflation rises; with the default policy rule output falls on impact.",
        ),
    )
    default_params = {"beta": 0.99, "sigma": 1.0, "kappa": 0.1, "phi_pi": 1.5, "phi_x": 0.125}

    def compute_steady_state(self) -> dict[str, float]:
        """Steady state of a model written in deviations: every variable is 0.

        Inputs: none. Returns: dict[str, float] mapping each of the 8 variable names to 0.0.
        """
        return {v.name: 0.0 for v in self.variables}

    def residuals(self, U, Z):
        """The three NK equations on whole paths.

        Inputs
        ------
        U : dict[str, np.ndarray]  candidate paths with keys 'x', 'pi', 'i'; float64, shape (T,).
        Z : dict[str, np.ndarray]  input paths with keys 'r_n', 'u', 'eps_i', 'wedge_pi'; float64, shape (T,).

        Returns
        -------
        dict[str, np.ndarray] with float64 arrays of shape (T,); 0 means the equation holds. With
        pi_e(t) = pi(t+1) + wedge_pi(t) (expected inflation; pi after the horizon is 0):
          'is_curve'  x(t) - x(t+1) + (i(t) - pi_e(t) - r_n(t)) / sigma.
                      Demand: today's output gap equals tomorrow's minus the effect of the real rate (i - pi_e) being
                      above the natural rate r_n.
          'phillips'  pi(t) - beta * pi_e(t) - kappa * x(t) - u(t).
                      Pricing: inflation depends on expected inflation, on how hot the economy is, and on cost-push.
          'taylor'    i(t) - phi_pi * pi(t) - phi_x * x(t) - eps_i(t).
                      Policy: the central bank sets the interest rate in response to inflation and the output gap,
                      plus any surprise.
        """
        p = self.params
        x, pi, i = U["x"], U["pi"], U["i"]
        pi_e = lead(pi, 0.0) + Z["wedge_pi"]
        is_curve = x - lead(x, 0.0) + (i - pi_e - Z["r_n"]) / p["sigma"]
        phillips = pi - p["beta"] * pi_e - p["kappa"] * x - Z["u"]
        taylor = i - p["phi_pi"] * pi - p["phi_x"] * x - Z["eps_i"]
        return {"is_curve": is_curve, "phillips": phillips, "taylor": taylor}

    def outputs(self, U, Z):
        """All eight variable paths once x, pi and i solve the model.

        Inputs: U with keys 'x', 'pi', 'i' and Z with keys 'r_n', 'u', 'eps_i', 'wedge_pi'; float64, shape (T,).
        Returns: dict[str, np.ndarray] with keys r_n, u, eps_i, wedge_pi, x, pi, i, rr; each float64, shape (T,).
        The derived real rate is rr(t) = i(t) - pi(t+1).
        """
        rr = U["i"] - lead(U["pi"], 0.0)
        return {**Z, **U, "rr": rr}
