from .base import SOLVER_REGISTRY, Solution, Solver, build_solver, register_solver
from dgse_llm.registry import import_submodules

import_submodules(__name__)

__all__ = ["SOLVER_REGISTRY", "Solution", "Solver", "build_solver", "register_solver"]
