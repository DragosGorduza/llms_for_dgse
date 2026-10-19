"""Mapping functions: convert a number parsed from LLM text into model input units."""

from __future__ import annotations

from dgse_llm.registry import Registry

MAPPING_REGISTRY = Registry("mapping")
register_mapping = MAPPING_REGISTRY.register


@register_mapping("identity")
def identity(value: float) -> float:
    return value


@register_mapping("linear")
def linear(value: float, slope: float = 1.0, intercept: float = 0.0) -> float:
    """intercept + slope * value. Example: percent -> decimal wedge with slope=-0.01 flips the sign too."""
    return intercept + slope * value


@register_mapping("pct")
def pct(value: float, factor: float = 1.0) -> float:
    """value / 100 * factor (the notebook's `f`, with factor=0.5)."""
    return value / 100.0 * factor


def apply_mapping(name: str, value: float, params: dict[str, float]) -> float:
    return float(MAPPING_REGISTRY.get(name)(value, **params))
