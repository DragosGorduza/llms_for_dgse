from .base import (
    MODEL_REGISTRY,
    ClassicShock,
    Model,
    Variable,
    build_model,
    lag,
    lead,
    register_model,
)
from dgse_llm.registry import import_submodules

# Importing every module registers the models, so adding a model needs no edit here.
import_submodules(__name__)

__all__ = ["MODEL_REGISTRY", "ClassicShock", "Model", "Variable", "build_model", "lag", "lead", "register_model"]
