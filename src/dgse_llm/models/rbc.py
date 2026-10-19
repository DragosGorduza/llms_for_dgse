"""Real business cycle (RBC) model with elastic labour. The plain-language guide is rbc.md.

The economy in one paragraph: one representative household owns the capital stock, rents it and its own labour to a
firm, and decides each period how much to consume, how much to save (i.e. invest in new capital) and how many hours
to work. The firm turns capital and labour into output. Output is split between consumption and investment. The
only things that move the economy are the two exogenous inputs `Z` (productivity) and `beta` (patience); everything
else responds to them. Everyone knows the future path of the inputs from period 0 (perfect foresight).

Period convention: rates such as `r` and `delta` are per period; the defaults are calibrated as if a period were a
quarter, but the code does not enforce this.
"""

from __future__ import annotations

import numpy as np

from .base import ClassicShock, Model, Variable, lag, lead, register_model


@register_model("rbc")
class RBC(Model):
    """RBC model. Unknowns: capital `K` and hours `L`. Inputs: productivity `Z` and the discount factor `beta`.

    Equations (see `residuals`): the household's consumption-saving condition (`euler`) and its work-versus-leisure
    condition (`labor`). Output, consumption, investment, the return on capital and the wage follow from K, L and Z.
    Parameters (`default_params`, all float):
      alpha  capital share of income (0.33): how much of output is paid to capital rather than labour.
      delta  depreciation rate (0.025): fraction of the capital stock that wears out each period.
      beta   steady-state discount factor (0.99): patience of the household in the steady state.
      sigma  curvature of utility from consumption (1.0): higher = household dislikes uneven consumption more.
      phi    inverse Frisch elasticity of labour supply (1.0): higher = hours respond less to the wage.
    """

    name = "rbc"
    variables = (
        Variable(
            "Z", "exogenous", "index, equal to 1 in the steady state",
            "Total factor productivity (TFP): how much output the economy produces from a given amount of capital and "
            "labour. It is the 'technology' of the economy and is not explained by the model.",
            shockable=True, bounds=(0.05, 5.0),
            if_up="A productivity boom (new technology, efficiency gains): the same capital and labour produce more, so "
                  "output, wages and the return on capital rise, and households work and invest more.",
            if_down="A productivity slump (supply disruption, loss of efficiency): output, wages and the return on "
                    "capital fall, and households work and invest less.",
        ),
        Variable(
            "beta", "exogenous", "unitless, about 0.99 in the steady state",
            "Discount factor between period t and t+1: how much the household values one unit of utility next period "
            "relative to one unit today. Closer to 1 means more patient. It is a preference (a taste), not a price.",
            shockable=True, bounds=(0.5, 1.2),
            if_up="The household becomes more patient (or more worried about the future): it consumes less now and "
                  "saves and invests more, so capital builds up and future consumption rises.",
            if_down="The household becomes more impatient (a 'live for today' mood): it consumes more now and invests "
                    "less, so capital falls and future consumption is lower.",
        ),
        Variable(
            "K", "unknown", "quantity (units of output), end of period",
            "Capital stock: the machines, buildings and equipment owned by the household at the END of period t. It is "
            "what the firm uses in period t+1, so today's saving decides tomorrow's production.",
            if_up="More productive capacity next period: higher output and wages, but a lower return per unit of capital.",
            if_down="Less productive capacity next period: lower output and wages, but a higher return per unit of capital.",
        ),
        Variable(
            "L", "unknown", "quantity (hours), equal to 1 in the steady state",
            "Labour: hours worked by the household in period t. The household chooses it by comparing the wage with the "
            "value of leisure.",
            if_up="People work more: output is higher today. This happens when wages are high or when the household feels "
                  "poorer.",
            if_down="People work less: output is lower today. This happens when wages are low or when the household feels "
                    "richer.",
        ),
        Variable(
            "Y", "derived", "quantity (units of output)",
            "Output (GDP): everything produced in period t, which is also the total income of the economy. "
            "Y = Z * K(t-1)^alpha * L^(1-alpha).",
            if_up="A boom: the economy produces and earns more.",
            if_down="A recession: the economy produces and earns less.",
        ),
        Variable(
            "C", "derived", "quantity (units of output)",
            "Consumption: the part of output the household uses up in period t. C = Y + (1-delta)*K(t-1) - K(t), i.e. "
            "what is left after the household chooses next period's capital.",
            if_up="The household spends more today (it is richer, more impatient, or saving less).",
            if_down="The household spends less today (it is poorer, more patient, or saving more).",
        ),
        Variable(
            "I", "derived", "quantity (units of output)",
            "Investment: output used to build new capital in period t, I = K(t) - (1-delta)*K(t-1). In the steady state "
            "it exactly replaces depreciation, delta*K.",
            if_up="The capital stock is growing (typically good news about the future, or more patience).",
            if_down="The capital stock is shrinking or growing more slowly (bad news about the future, or more impatience).",
        ),
        Variable(
            "r", "derived", "rate per period, 0.01 = 1%",
            "Net real return on capital earned in period t: the rent paid per unit of capital minus depreciation. It is "
            "the reward for saving, so it guides the consumption-saving choice.",
            if_up="Saving is more rewarding (capital is scarce or more productive): households shift consumption into the "
                  "future.",
            if_down="Saving is less rewarding (capital is plentiful or less productive): households shift consumption "
                    "towards the present.",
        ),
        Variable(
            "w", "derived", "price (units of output per hour)",
            "Real wage: what the firm pays per hour of labour in period t, equal to the marginal product of labour.",
            if_up="Work pays more: households supply more hours and earn more.",
            if_down="Work pays less: households supply fewer hours and earn less.",
        ),
    )
    # euler: consumption-saving condition. labor: work-versus-leisure condition.
    targets = ("euler", "labor")
    # Ready-made shocks that need no LLM. Reference them in a config with `source: classic, classic: <name>`.
    classic_shocks = (
        ClassicShock(
            "tfp_boom_persistent", "Z", 0.01, "geometric_decay", rho=0.9,
            description="A 1% productivity boom that fades slowly (about 10% of it is left after 20 periods). The textbook "
                        "technology shock: output, hours, consumption and investment all rise, then return to normal.",
        ),
        ClassicShock(
            "tfp_slump_one_time", "Z", -0.02, "one_time",
            description="A one-period 2% productivity drop, for example a short supply disruption. The economy falls and "
                        "then recovers on its own through capital and hours.",
        ),
        ClassicShock(
            "tfp_slump_multi_period", "Z", -0.01, "multi_period", duration=4,
            description="A 1% productivity drop lasting 4 periods (a year), then an abrupt return to normal. Households "
                        "see the end coming from period 0 and smooth their consumption.",
        ),
        ClassicShock(
            "impatience_wave", "beta", -0.005, "multi_period", duration=4,
            description="Households become less patient for 4 periods (discount factor falls by 0.005): they consume more "
                        "and invest less, so capital is lower afterwards.",
        ),
    )
    default_params = {"alpha": 0.33, "delta": 0.025, "beta": 0.99, "sigma": 1.0, "phi": 1.0}

    def compute_steady_state(self) -> dict[str, float]:
        """Closed-form steady state: all variables constant over time, nothing shocked.

        Inputs: none (reads `self.params`, dict[str, float]).
        Returns: dict[str, float] with one scalar for each of Z, beta, K, L, Y, C, I, r, w.
        Side effect: sets `self.chi` (float), the weight on the disutility of work, chosen so that steady-state hours
        L equal 1. It is needed by `residuals`.
        """
        p = self.params
        a, d = p["alpha"], p["delta"]
        r = 1.0 / p["beta"] - 1.0
        L = 1.0
        K = (a / (r + d)) ** (1.0 / (1.0 - a)) * L
        Y = K**a * L ** (1.0 - a)
        w = (1.0 - a) * Y / L
        C = Y - d * K
        self.chi = w * C ** (-p["sigma"]) / L ** p["phi"]
        return {"Z": 1.0, "beta": p["beta"], "K": K, "L": L, "Y": Y, "C": C, "I": d * K, "r": r, "w": w}

    def _core(self, U, Z):
        """Production side and resource constraint, shared by `residuals` and `outputs`.

        Inputs
        ------
        U : dict[str, np.ndarray]  needs keys 'K' and 'L'; float64 arrays of shape (T,).
        Z : dict[str, np.ndarray]  needs key 'Z'; float64 array of shape (T,).

        Returns
        -------
        tuple of five float64 arrays, each of shape (T,), in this order:
          K_lag  capital at the start of each period (K shifted by one; the steady state before t=0)
          Y      output
          r      net return on capital
          w      real wage
          C      consumption, from the resource constraint
        """
        p, ss = self.params, self.steady_state
        a, d = p["alpha"], p["delta"]
        K, L, z = U["K"], U["L"], Z["Z"]
        K_lag = lag(K, ss["K"])
        Y = z * K_lag**a * L ** (1.0 - a)
        r = a * z * (K_lag / L) ** (a - 1.0) - d
        w = (1.0 - a) * z * (K_lag / L) ** a
        C = Y + (1.0 - d) * K_lag - K
        return K_lag, Y, r, w, C

    def residuals(self, U, Z):
        """The two household conditions that pin down capital and hours.

        Inputs
        ------
        U : dict[str, np.ndarray]  candidate paths with keys 'K' and 'L'; float64, shape (T,).
        Z : dict[str, np.ndarray]  input paths with keys 'Z' and 'beta'; float64, shape (T,).

        Returns
        -------
        dict[str, np.ndarray] with float64 arrays of shape (T,); 0 means the condition holds:
          'euler'  C(t)^-sigma  -  beta(t) * C(t+1)^-sigma * (1 + r(t+1)).
                   In words: giving up one unit of consumption today and consuming the proceeds tomorrow must leave
                   the household exactly as well off. If it is positive, the household would rather consume more now
                   (save less).
          'labor'  chi * L(t)^phi  -  w(t) * C(t)^-sigma.
                   In words: the pain of one more hour of work must equal the extra consumption the wage buys.
                   If it is positive, the household would rather work less.
        """
        p, ss = self.params, self.steady_state
        s, phi = p["sigma"], p["phi"]
        _, _, r, w, C = self._core(U, Z)
        euler = C ** (-s) - Z["beta"] * lead(C, ss["C"]) ** (-s) * (1.0 + lead(r, ss["r"]))
        labor = self.chi * U["L"] ** phi - w * C ** (-s)
        return {"euler": euler, "labor": labor}

    def outputs(self, U, Z):
        """All nine variable paths once K and L solve the model.

        Inputs: U with keys 'K', 'L' and Z with keys 'Z', 'beta'; float64 arrays of shape (T,).
        Returns: dict[str, np.ndarray] with keys Z, beta, K, L, Y, C, I, r, w; each float64, shape (T,).
        Investment is computed here as I = K(t) - (1-delta) * K(t-1).
        """
        K_lag, Y, r, w, C = self._core(U, Z)
        I = U["K"] - (1.0 - self.params["delta"]) * K_lag
        return {**Z, **U, "Y": Y, "C": C, "I": I, "r": r, "w": w}
