"""Model interface: a macro model is a set of equations over whole time paths, independent of any solver.

Vocabulary used in every model file
-----------------------------------
* T          : int. The number of periods simulated (the horizon). Period t runs from 0 to T-1.
* A "path"   : a 1-D numpy array of dtype float64 and shape (T,), one number per period.
* Z          : dict[str, np.ndarray]. The EXOGENOUS paths: inputs the model does not explain (e.g. TFP). These are
               the variables an LLM-generated shock may modify. Keys are the names of the exogenous variables.
* U          : dict[str, np.ndarray]. The UNKNOWN paths: variables the solver must find (e.g. capital). Keys are
               the names of the unknown variables.
* residuals  : the equations of the model written as "left side minus right side". A path of unknowns solves the
               model when every residual is zero in every period.
* steady state: the long-run resting point of the economy when nothing is shocked. Before t=0 the economy is
               assumed to sit at the steady state, and after the horizon it returns to it, so a lag at t=0 and a
               lead at t=T-1 both use the steady-state value.
* perfect foresight: households and firms know the whole future path of Z from t=0 (anticipated shocks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np

from dgse_llm.registry import Registry
from dgse_llm.schemas import ModelConfig, TimeProfile

Role = Literal["exogenous", "unknown", "derived"]

MODEL_REGISTRY = Registry("model")
register_model = MODEL_REGISTRY.register


@dataclass(frozen=True)
class Variable:
    """Documentation and metadata of one model variable.

    Attributes
    ----------
    name : str
        Key used in every dict of paths (U, Z, solver output) and in shock configs.
    role : {"exogenous", "unknown", "derived"}
        exogenous = an input path given from outside (an LLM shock can modify it if `shockable`);
        unknown   = solved for by the solver;
        derived   = computed afterwards from the other two kinds, only reported.
    units : str
        Plain-language unit, e.g. "index, 1 at steady state" or "decimal per quarter (0.01 = 1%)".
    description : str
        What the variable is and its macroeconomic interpretation, in plain language.
    shockable : bool
        True if an experiment may add to / replace this input. Only exogenous variables can be shockable.
    bounds : tuple[float, float] | None
        (lowest, highest) allowed LEVEL of the input (steady state plus shock). A shock path leaving this range is
        rejected, or clipped when the shock config says bounds_policy: clip. Only checked for shockable inputs.
    if_up : str
        What it means for the economy when this variable is higher than its steady-state value.
    if_down : str
        What it means for the economy when this variable is lower than its steady-state value.
    """

    name: str
    role: Role
    units: str
    description: str
    shockable: bool = False
    bounds: tuple[float, float] | None = None
    if_up: str = ""
    if_down: str = ""


@dataclass(frozen=True)
class ClassicShock:
    """A named, ready-made shock for one input of a model: the 'classic' way to generate a scenario.

    Classic shocks need no LLM. An experiment refers to one by name (`source: classic`, `classic: <name>`), may scale
    it with `intensity`, and may combine it with an LLM shock to compare the two.

    Attributes
    ----------
    name : str
        Identifier used in configs, e.g. "tfp_slump_one_time".
    variable : str
        The exogenous, shockable input that is modified.
    size : float
        Change in the input, in the input's own units, at intensity 1 (negative = the input falls). Added to the
        baseline value of the input in every active period.
    kind : {"one_time", "multi_period", "geometric_decay"}
        Timing. one_time: one period, then the model's own dynamics take over. multi_period: the same size for
        `duration` periods. geometric_decay: starts at `size` and shrinks by the factor `rho` each period.
    duration : int
        Number of periods for kind multi_period.
    rho : float
        Persistence for kind geometric_decay (0 = gone after one period, close to 1 = very long-lived).
    description : str
        The story in plain language: what real-world event this stands for and what to expect.
    """

    name: str
    variable: str
    size: float
    kind: Literal["one_time", "multi_period", "geometric_decay"]
    description: str
    duration: int = 1
    rho: float = 0.9

    def profile(self) -> TimeProfile:
        """The timing as a `TimeProfile` starting at period 0."""
        return TimeProfile(kind=self.kind, start=0, duration=self.duration, rho=self.rho)


def lead(x: np.ndarray, ss: float) -> np.ndarray:
    """Next-period value x_{t+1} of a path.

    Parameters
    ----------
    x : np.ndarray, float64, shape (T,)  the path.
    ss : float                           steady-state value, used for the period after the last one.

    Returns
    -------
    np.ndarray, float64, shape (T,)  element t is x[t+1]; the last element is `ss`.
    """
    return np.append(x[1:], ss)


def lag(x: np.ndarray, ss: float) -> np.ndarray:
    """Previous-period value x_{t-1} of a path.

    Parameters
    ----------
    x : np.ndarray, float64, shape (T,)  the path.
    ss : float                           steady-state value, used for the period before t=0.

    Returns
    -------
    np.ndarray, float64, shape (T,)  element t is x[t-1]; the first element is `ss`.
    """
    return np.insert(x[:-1], 0, ss)


class Model(ABC):
    """Base class of every macro model. Subclasses fill in the class attributes and three methods.

    Class attributes a subclass must define
    ---------------------------------------
    name : str                         registry name, equal to the .py file name and the `name:` of its config.
    variables : tuple[Variable, ...]   every variable, each with a plain-language description (see `Variable`).
    targets : tuple[str, ...]          names of the equation residuals returned by `residuals`; exactly one per unknown.
    default_params : dict[str, float]  every parameter and its default value; a config may override any of them.
    classic_shocks : tuple[ClassicShock, ...]  ready-made shocks that need no LLM (several, one-time and multi-period).

    Methods a subclass must implement: `compute_steady_state`, `residuals`; and should override `outputs`
    if it has derived variables.
    """

    name: ClassVar[str] = ""
    variables: ClassVar[tuple[Variable, ...]] = ()
    targets: ClassVar[tuple[str, ...]] = ()
    default_params: ClassVar[dict[str, float]] = {}
    classic_shocks: ClassVar[tuple[ClassicShock, ...]] = ()

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """Parameters
        ----------
        params : dict[str, float] | None
            Overrides of `default_params` (e.g. {"alpha": 0.3}). Unknown names raise ValueError.
            The merged values are stored in `self.params` (dict[str, float]).
        """
        params = dict(params or {})
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"Unknown parameters for model '{self.name}': {sorted(unknown)}. "
                f"Valid: {sorted(self.default_params)}"
            )
        self.params: dict[str, float] = {**self.default_params, **params}
        self._ss: dict[str, float] | None = None

    @property
    def unknowns(self) -> list[str]:
        """Names (list[str]) of the variables with role 'unknown', in declaration order."""
        return [v.name for v in self.variables if v.role == "unknown"]

    @property
    def exogenous(self) -> list[str]:
        """Names (list[str]) of the variables with role 'exogenous' (the inputs), in declaration order."""
        return [v.name for v in self.variables if v.role == "exogenous"]

    @property
    def shockable(self) -> list[str]:
        """Names (list[str]) of the inputs an experiment may shock."""
        return [v.name for v in self.variables if v.shockable]

    def variable(self, name: str) -> Variable:
        """The `Variable` documentation object called `name`. Raises KeyError if there is none."""
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"Model '{self.name}' has no variable '{name}'")

    def classic_shock(self, name: str) -> ClassicShock:
        """The `ClassicShock` called `name`. Raises KeyError listing the available names if there is none."""
        for shock in self.classic_shocks:
            if shock.name == name:
                return shock
        raise KeyError(
            f"Model '{self.name}' has no classic shock '{name}'. Available: {[s.name for s in self.classic_shocks]}"
        )

    @property
    def steady_state(self) -> dict[str, float]:
        """Steady-state value of every variable, dict[str, float]. Computed once, then cached."""
        if self._ss is None:
            ss = self.compute_steady_state()
            missing = [v.name for v in self.variables if v.name not in ss]
            if missing:
                raise RuntimeError(f"compute_steady_state of '{self.name}' misses {missing}")
            self._ss = ss
        return self._ss

    def cache_key(self) -> tuple:
        """Hashable identity (model name plus sorted parameters), used to cache Jacobians and baselines."""
        return (self.name, tuple(sorted(self.params.items())))

    def baseline_exog(self, T: int) -> dict[str, np.ndarray]:
        """The unshocked inputs: every exogenous variable held at its steady-state value.

        Parameters
        ----------
        T : int  number of periods.

        Returns
        -------
        dict[str, np.ndarray]  one float64 array of shape (T,) per exogenous variable.
        """
        return {n: np.full(T, self.steady_state[n], dtype=float) for n in self.exogenous}

    @abstractmethod
    def compute_steady_state(self) -> dict[str, float]:
        """Solve for the resting point of the economy.

        Returns
        -------
        dict[str, float]  one scalar for EVERY variable in `variables` (inputs, unknowns and derived ones).
        """

    @abstractmethod
    def residuals(self, U: dict[str, np.ndarray], Z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Evaluate the model equations on whole paths.

        Parameters
        ----------
        U : dict[str, np.ndarray]  candidate paths of the unknowns; each float64, shape (T,).
        Z : dict[str, np.ndarray]  the exogenous input paths; each float64, shape (T,).

        Returns
        -------
        dict[str, np.ndarray]  keys exactly `targets`; each float64, shape (T,). A value of 0 in period t means the
        equation holds in period t. The solver changes U until all residuals are 0.
        """

    def outputs(self, U: dict[str, np.ndarray], Z: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Every variable's path once U solves the model, including derived variables.

        Parameters and shapes are as in `residuals`. Returns dict[str, np.ndarray] with one float64 array of shape
        (T,) per entry of `variables` (keys equal the variable names). The default returns only inputs and unknowns;
        override it to add derived variables.
        """
        return {**Z, **U}


def build_model(config: ModelConfig) -> Model:
    return MODEL_REGISTRY.get(config.name)(config.params)
