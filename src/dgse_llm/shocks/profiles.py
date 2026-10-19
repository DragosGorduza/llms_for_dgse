"""Time profiles: unit-height shapes over the horizon. Intensity is applied separately."""

from __future__ import annotations

from typing import Callable

import numpy as np

from dgse_llm.registry import Registry
from dgse_llm.schemas import TimeProfile

PROFILE_REGISTRY = Registry("time profile")
register_profile = PROFILE_REGISTRY.register

ProfileFn = Callable[[int, TimeProfile], np.ndarray]


@register_profile("one_time")
def one_time(T: int, p: TimeProfile) -> np.ndarray:
    """Impulse at `start`; afterwards the model's own dynamics take over."""
    shape = np.zeros(T)
    if p.start < T:
        shape[p.start] = 1.0
    return shape


@register_profile("multi_period")
def multi_period(T: int, p: TimeProfile) -> np.ndarray:
    """Constant at 1 for `duration` periods from `start`."""
    shape = np.zeros(T)
    shape[p.start : p.start + p.duration] = 1.0
    return shape


@register_profile("geometric_decay")
def geometric_decay(T: int, p: TimeProfile) -> np.ndarray:
    """rho ** (t - start) from `start` on."""
    shape = np.zeros(T)
    if p.start < T:
        shape[p.start :] = p.rho ** np.arange(T - p.start)
    return shape


@register_profile("from_series")
def from_series(T: int, p: TimeProfile) -> np.ndarray:
    """The given series placed at `start`, truncated at the horizon."""
    shape = np.zeros(T)
    series = np.asarray(p.series or [], dtype=float)
    n = max(0, min(len(series), T - p.start))
    shape[p.start : p.start + n] = series[:n]
    return shape


def make_shape(T: int, profile: TimeProfile) -> np.ndarray:
    if profile.kind == "narrative_timeline":
        raise ValueError("narrative_timeline has no fixed shape; it is built from LLM answers")
    return PROFILE_REGISTRY.get(profile.kind)(T, profile)
