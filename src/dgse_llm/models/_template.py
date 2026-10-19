"""Template for a new model. Copy to <name>.py (no leading underscore), rename, fill in, and write <name>.md.

Underscore-prefixed modules are not auto-imported, so this file registers nothing.

Documentation rules (enforced by tests/models/test_conformance.py)
-------------------------------------------------------------------
* Every Variable needs a plain-language `description` (what it is and its macro interpretation), plus `if_up` and
  `if_down` (what it means for the economy when the variable is above / below its steady state), and `units`.
* The .md file needs one section per variable and per target; see rbc.md for the format.
* Every method docstring states the dtype and shape of each input and output (see base.py for the vocabulary:
  T, path, U, Z, residuals).
"""

from __future__ import annotations

import numpy as np

from .base import ClassicShock, Model, Variable, lag, lead, register_model


@register_model("my_model")
class MyModel(Model):
    """One paragraph: what economy this is, who the agents are, what moves it. List each parameter in `default_params`
    with its meaning in words."""

    name = "my_model"
    variables = (
        # Exogenous inputs the LLM may modify: shockable=True and level bounds.
        Variable(
            "Z", "exogenous", "index, 1 in the steady state",
            "Productivity: how much output the economy gets from its capital.",
            shockable=True, bounds=(0.05, 5.0),
            if_up="Output and the return on capital rise.",
            if_down="Output and the return on capital fall.",
        ),
        # One unknown per entry in `targets`.
        Variable(
            "K", "unknown", "quantity", "Capital stock at the end of the period.",
            if_up="More productive capacity next period.",
            if_down="Less productive capacity next period.",
        ),
        Variable(
            "Y", "derived", "quantity", "Output (GDP).",
            if_up="A boom.",
            if_down="A recession.",
        ),
    )
    targets = ("my_equation",)
    # At least three ready-made shocks that need no LLM, including one-time and multi-period ones. Describe each in
    # plain language here and in the .md (a table row and a '### Classic shock <name>' section).
    classic_shocks = (
        ClassicShock("tfp_slump_one_time", "Z", -0.02, "one_time",
                     description="A one-period 2% productivity drop (short supply disruption)."),
        ClassicShock("tfp_slump_multi_period", "Z", -0.01, "multi_period", duration=4,
                     description="A 1% productivity drop lasting 4 periods, then back to normal."),
        ClassicShock("tfp_boom_persistent", "Z", 0.01, "geometric_decay", rho=0.9,
                     description="A 1% productivity boom that fades slowly (10% left after about 20 periods)."),
    )
    default_params = {"alpha": 0.33}

    def compute_steady_state(self) -> dict[str, float]:
        """Inputs: none (reads self.params). Returns dict[str, float]: one scalar for EVERY variable above."""
        return {"Z": 1.0, "K": 1.0, "Y": 1.0}

    def residuals(self, U: dict[str, np.ndarray], Z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Inputs: U['K'] and Z['Z'], float64 arrays of shape (T,). Returns {'my_equation': float64 array (T,)}, 0 when
        capital equals Z * K(t-1)^alpha. Use lead(x, ss) / lag(x, ss) for time shifts."""
        ss = self.steady_state
        K_lag = lag(U["K"], ss["K"])
        return {"my_equation": U["K"] - Z["Z"] * K_lag**self.params["alpha"]}

    def outputs(self, U, Z):
        """Inputs as in `residuals`. Returns dict[str, np.ndarray] with keys Z, K, Y; each float64, shape (T,)."""
        K_lag = lag(U["K"], self.steady_state["K"])
        return {**Z, **U, "Y": Z["Z"] * K_lag ** self.params["alpha"]}
