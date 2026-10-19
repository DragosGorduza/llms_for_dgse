"""Krusell-Smith model (heterogeneous households, one asset), plain numpy. Plain-language guide: krusell_smith.md.

The economy in one paragraph: many households (a continuum) differ in two ways: their income this period (a random
'skill' or employment shock that changes over time) and how much they have saved. They cannot borrow (assets must stay
above `amin`), so they save as a buffer against bad income draws ('precautionary saving'). All savings are invested in
capital, which a single representative firm combines with labour to produce output. Because households differ, the
reaction of aggregate consumption to a change in interest rates, wages or taste depends on how many households are
close to their borrowing limit. That is the main difference from the RBC model, where one household stands for
everyone. Labour supply is fixed at 1 (people do not choose hours). Everyone knows the future path of the inputs
from period 0 (perfect foresight), and the distribution of households starts at its steady-state shape in period 0.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ._het_agents import (
    backward_step,
    forward_step,
    lottery,
    make_agrid,
    rouwenhorst,
    solve_household_ss,
)
from .base import ClassicShock, Model, Variable, lag, register_model


@register_model("krusell_smith")
class KrusellSmith(Model):
    """Krusell-Smith economy. One unknown: aggregate capital `K`. Inputs: `Z`, `d`, `beta`.

    The single equation (`asset_mkt`) says households' total savings must equal the capital the firm uses.
    Parameters (`default_params`, all float; integers are stored as float and cast where needed):
      alpha       capital share of income (0.11).
      delta       depreciation rate per period (0.025).
      eis         elasticity of intertemporal substitution (1.0): how willing households are to move consumption
                  across time when the interest rate changes; 1 = log utility.
      rho_e       persistence of the income shock (0.966): closer to 1 = income changes are long-lasting.
      sd_e        cross-sectional standard deviation of log income (0.5): how unequal incomes are.
      nE          number of income levels on the grid (7).
      nA          number of asset levels on the grid (100); more points = more accurate and slower.
      amin, amax  lowest (borrowing limit, 0 = no borrowing) and highest asset level on the grid (0 and 200).
      agrid_curv  how strongly grid points are packed near amin (1000).
      target_r    steady-state net return on capital that the discount factor is calibrated to hit (0.01).
      target_Y    steady-state output that fixes the level of productivity Z (1.0).
      ss_tol      convergence tolerance of the steady-state iterations (1e-11).
      beta_lo, beta_hi  search bracket for the calibrated discount factor (0.93, 0.986).
    """

    name = "krusell_smith"
    variables = (
        Variable(
            "Z", "exogenous", "index, about 0.88 in the steady state (calibrated so that output is 1)",
            "Total factor productivity (TFP): how much output the firm gets from a given amount of capital and labour.",
            shockable=True, bounds=(0.05, 5.0),
            if_up="A productivity boom: output, wages and the return on capital rise; households earn more and save more.",
            if_down="A productivity slump: output, wages and the return on capital fall; households earn less and save "
                    "less.",
        ),
        Variable(
            "d", "exogenous", "marginal-utility units, 0 in the steady state (marginal utility of consumption is about 1)",
            "Taste shifter / 'desire to postpone spending'. It is subtracted from the satisfaction of consuming today: "
            "marginal utility is c^(-1/eis) - d. It captures behaviour that changes how much households want to spend "
            "for non-economic reasons (fear, lockdowns, caution, or pent-up urge to spend) without changing prices or "
            "incomes. This is the input the notebook's pandemic narrative moves; d = 0.02 is roughly a 2% change in the "
            "value of spending today.",
            shockable=True, bounds=(-0.2, 0.2),
            if_up="Households want to spend less today (e.g. voluntary distancing or fear): consumption falls on impact "
                  "and savings rise, so capital rises and the interest rate falls.",
            if_down="Households want to spend more today (e.g. pent-up demand or confidence): consumption rises on "
                    "impact and savings fall, so capital falls and the interest rate rises.",
        ),
        Variable(
            "beta", "exogenous", "unitless, about 0.98 in the steady state (calibrated)",
            "Discount factor between period t and t+1: how patient households are. Closer to 1 = more patient.",
            shockable=True, bounds=(0.5, 1.0),
            if_up="Households become more patient: they save more and consume less today, so capital builds up.",
            if_down="Households become less patient: they save less and consume more today, so capital falls.",
        ),
        Variable(
            "K", "unknown", "quantity (units of output), end of period",
            "Aggregate capital stock at the end of period t. In equilibrium it equals total household savings `A`. The "
            "firm uses it in period t+1.",
            if_up="More productive capacity next period: higher output and wages, lower return on capital.",
            if_down="Less productive capacity next period: lower output and wages, higher return on capital.",
        ),
        Variable(
            "Y", "derived", "quantity (units of output), 1 in the steady state",
            "Output (GDP) in period t: Z * K(t-1)^alpha, since labour is fixed at 1.",
            if_up="A boom: more production and income.",
            if_down="A recession: less production and income.",
        ),
        Variable(
            "r", "derived", "rate per period, 0.01 = 1% in the steady state",
            "Net real return on capital in period t: the reward for saving. It is high when capital is scarce or "
            "productive.",
            if_up="Saving is more rewarding: households save more and consume less.",
            if_down="Saving is less rewarding: households save less and consume more.",
        ),
        Variable(
            "w", "derived", "price (units of output per unit of labour), 0.89 in the steady state",
            "Real wage in period t: the pay per unit of labour. Households' income is w times their individual "
            "productivity level.",
            if_up="Households earn more labour income, especially those living hand to mouth: aggregate consumption rises.",
            if_down="Households earn less labour income: aggregate consumption falls.",
        ),
        Variable(
            "A", "derived", "quantity (units of output)",
            "Aggregate household assets chosen at the end of period t: the sum of everyone's savings. The model "
            "requires A = K.",
            if_up="Households are saving more in total, which pushes capital up.",
            if_down="Households are saving less in total, which pushes capital down.",
        ),
        Variable(
            "C", "derived", "quantity (units of output)",
            "Aggregate consumption in period t: the sum of what all households spend.",
            if_up="Households spend more in total.",
            if_down="Households spend less in total.",
        ),
        Variable(
            "goods_mkt", "derived", "quantity (units of output), 0 in the steady state",
            "Goods-market check Y - C - (K(t) - (1-delta)*K(t-1)): output minus consumption minus investment. It is "
            "not an economic variable but a consistency test (Walras' law): when the asset market clears, this must be "
            "0 in every period.",
            if_up="A positive value signals a numerical error: output exceeds consumption plus investment.",
            if_down="A negative value signals a numerical error: consumption plus investment exceed output.",
        ),
    )
    # asset_mkt: household savings minus the capital stock.
    targets = ("asset_mkt",)
    # Ready-made shocks that need no LLM. Reference them in a config with `source: classic, classic: <name>`.
    classic_shocks = (
        ClassicShock(
            "tfp_boom_persistent", "Z", 0.01, "geometric_decay", rho=0.8,
            description="A 1% productivity boom that fades quickly (20% of it is left after 7 periods). Output, wages and "
                        "consumption rise; households save part of the gain, so capital rises with a lag.",
        ),
        ClassicShock(
            "tfp_slump_one_time", "Z", -0.02, "one_time",
            description="A one-period 2% productivity drop. Wages and output fall; households with little savings cut "
                        "consumption most.",
        ),
        ClassicShock(
            "precautionary_saving", "d", 0.02, "multi_period", duration=6,
            description="Households want to spend less (fear, caution) for 6 periods: the taste shifter d rises by 0.02. "
                        "Consumption falls and saving rises. This is the classic counterpart of the LLM pandemic shock.",
        ),
        ClassicShock(
            "patience_burst", "beta", 0.002, "one_time",
            description="A one-period rise of 0.002 in the discount factor: households save more and consume less for one "
                        "period, so capital rises.",
        ),
    )
    default_params = {
        "alpha": 0.11,
        "delta": 0.025,
        "eis": 1.0,
        "rho_e": 0.966,
        "sd_e": 0.5,
        "nE": 7,
        "nA": 100,
        "amin": 0.0,
        "amax": 200.0,
        "agrid_curv": 1000.0,
        "target_r": 0.01,
        "target_Y": 1.0,
        "ss_tol": 1e-11,
        "beta_lo": 0.93,
        "beta_hi": 0.986,
    }

    def compute_steady_state(self) -> dict[str, float]:
        """Calibrate and solve the steady state (all variables constant, distribution of households stationary).

        Steps: (1) from target_r and target_Y get capital K, productivity Z and the wage w in closed form;
        (2) search (Brent's method) for the discount factor beta at which households' total savings equal K;
        (3) store the household policy functions and distribution in `self._hh` for `_simulate`.

        Inputs: none (reads `self.params`).
        Returns: dict[str, float], one scalar for each of Z, d, beta, K, Y, r, w, A, C, goods_mkt.
        Raises RuntimeError if beta_lo / beta_hi do not bracket the solution.
        Side effect: sets `self._hh`, a dict with
          'a_grid' float64 (nA,)      asset grid
          'e_grid' float64 (nE,)      income levels, mean 1
          'Pi'     float64 (nE, nE)   income transition matrix; Pi[i, j] = P(next income j | income i), rows sum to 1
          'Va'     float64 (nE, nA)   marginal value of an extra unit of assets (steady state)
          'D'      float64 (nE, nA)   stationary share of households at each (income, assets) point; sums to 1
        """
        p = self.params
        alpha, delta, eis, tol = p["alpha"], p["delta"], p["eis"], p["ss_tol"]
        e_grid, pi_e, Pi = rouwenhorst(p["rho_e"], p["sd_e"], int(p["nE"]))
        a_grid = make_agrid(p["amin"], p["amax"], int(p["nA"]), p["agrid_curv"])
        r, Y = p["target_r"], p["target_Y"]
        # With L = 1: r + delta = alpha Y / K and Y = Z K^alpha.
        K = alpha * Y / (r + delta)
        Z = Y / K**alpha
        w = (1.0 - alpha) * Y

        warm: dict = {}

        def excess_assets(beta: float) -> float:
            """Households' total savings minus capital K at discount factor `beta` (float); 0 at the solution."""
            Va, a, c, D = solve_household_ss(
                a_grid, e_grid, Pi, pi_e, r, w, beta, eis, tol, warm.get("Va"), warm.get("D")
            )
            warm.update(Va=Va, D=D, a=a, c=c)
            return float(np.sum(D * a) - K)

        try:
            beta = brentq(excess_assets, p["beta_lo"], p["beta_hi"], xtol=1e-12, rtol=1e-12, maxiter=60)
        except ValueError as exc:
            raise RuntimeError(
                "Could not bracket the discount factor; adjust params beta_lo / beta_hi"
            ) from exc
        # Re-solve cold at the root so the stored objects do not depend on the warm-start path.
        Va, a, c, D = solve_household_ss(a_grid, e_grid, Pi, pi_e, r, w, beta, eis, tol)
        self._hh = {"a_grid": a_grid, "e_grid": e_grid, "Pi": Pi, "Va": Va, "D": D}
        C = float(np.sum(D * c))
        return {
            "Z": Z, "d": 0.0, "beta": beta, "K": K, "Y": Y, "r": r, "w": w,
            "A": float(np.sum(D * a)), "C": C, "goods_mkt": Y - C - delta * K,
        }

    def _simulate(self, U, Z):
        """Run the household side of the economy through time for a given capital path.

        Steps: (1) from the capital path and Z compute the interest rate and wage each period; (2) go BACKWARDS in
        time from the steady-state value function to get every period's savings and consumption rules; (3) go
        FORWARDS from the steady-state distribution of households to get aggregate savings and consumption.

        Inputs
        ------
        U : dict[str, np.ndarray]  needs 'K'; float64, shape (T,).
        Z : dict[str, np.ndarray]  needs 'Z', 'd', 'beta'; float64, shape (T,).

        Returns
        -------
        dict[str, np.ndarray], each float64 of shape (T,):
          'K_lag' capital at the start of each period; 'r' return on capital; 'w' wage; 'Y' output;
          'A' total household savings chosen in the period; 'C' total consumption.
        """
        p, ss = self.params, self.steady_state
        h = self._hh
        alpha, delta, eis = p["alpha"], p["delta"], p["eis"]
        K = U["K"]
        T = len(K)
        K_lag = lag(K, ss["K"])
        r = alpha * Z["Z"] * K_lag ** (alpha - 1.0) - delta
        w = (1.0 - alpha) * Z["Z"] * K_lag**alpha

        policies = [None] * T
        Va_next = h["Va"]
        for t in range(T - 1, -1, -1):
            Va, a, c = backward_step(
                h["Pi"] @ Va_next, h["a_grid"], h["e_grid"], r[t], w[t], Z["beta"][t], eis, Z["d"][t]
            )
            idx, w_hi = lottery(a, h["a_grid"])
            policies[t] = (a, c, idx, w_hi)
            Va_next = Va

        D = h["D"]
        A = np.empty(T)
        C = np.empty(T)
        for t in range(T):
            a, c, idx, w_hi = policies[t]
            A[t] = np.sum(D * a)
            C[t] = np.sum(D * c)
            D = forward_step(D, idx, w_hi, h["Pi"])
        Y = Z["Z"] * K_lag**alpha
        return {"K_lag": K_lag, "r": r, "w": w, "A": A, "C": C, "Y": Y}

    def residuals(self, U, Z):
        """The asset-market clearing condition.

        Inputs: U with key 'K' and Z with keys 'Z', 'd', 'beta'; float64 arrays of shape (T,).
        Returns: dict with one key 'asset_mkt', float64 array of shape (T,): total household savings A(t) minus
        capital K(t). 0 means households save exactly the capital the firm needs. A positive value means households
        want to save more than there is capital (so the interest rate must fall).
        """
        sim = self._simulate(U, Z)
        return {"asset_mkt": sim["A"] - U["K"]}

    def outputs(self, U, Z):
        """All ten variable paths once K solves the model.

        Inputs: U with key 'K' and Z with keys 'Z', 'd', 'beta'; float64 arrays of shape (T,).
        Returns: dict[str, np.ndarray] with keys Z, d, beta, K, Y, r, w, A, C, goods_mkt; each float64, shape (T,).
        """
        sim = self._simulate(U, Z)
        K = U["K"]
        goods = sim["Y"] - sim["C"] - (K - (1.0 - self.params["delta"]) * sim["K_lag"])
        return {
            **Z, **U, "Y": sim["Y"], "r": sim["r"], "w": sim["w"],
            "A": sim["A"], "C": sim["C"], "goods_mkt": goods,
        }
