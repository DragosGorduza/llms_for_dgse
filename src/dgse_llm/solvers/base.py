"""Solver interface. A solver finds unknown paths that zero a model's residuals for given exogenous paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from dgse_llm.models import Model
from dgse_llm.registry import Registry
from dgse_llm.schemas import Diagnostics

SOLVER_REGISTRY = Registry("solver")
register_solver = SOLVER_REGISTRY.register


@dataclass
class Solution:
    paths: dict[str, np.ndarray]
    diagnostics: Diagnostics


class Solver(ABC):
    name: ClassVar[str] = ""

    def __init__(self, **options: Any) -> None:
        self.options = options

    @abstractmethod
    def solve(self, model: Model, exog: dict[str, np.ndarray], T: int) -> Solution:
        """Return the path of every model variable (exogenous, unknown and derived), each of length T."""


def build_solver(name: str, options: dict[str, Any] | None = None) -> Solver:
    return SOLVER_REGISTRY.get(name)(**(options or {}))
