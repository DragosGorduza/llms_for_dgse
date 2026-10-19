"""Summary statistics of impulse responses (shocked minus baseline)."""

from __future__ import annotations

import numpy as np


def irf_metrics(irf: np.ndarray) -> dict[str, float | int | None]:
    """peak (signed), peak_t, cumulative (sum), half_life (periods after the peak to fall to half, else None)."""
    irf = np.asarray(irf, dtype=float)
    mag = np.abs(irf)
    peak_t = int(np.argmax(mag))
    if mag[peak_t] == 0.0:
        return {"peak": 0.0, "peak_t": 0, "cumulative": 0.0, "half_life": None}
    half_life = None
    for t in range(peak_t, len(irf)):
        if mag[t] <= mag[peak_t] / 2.0:
            half_life = t - peak_t
            break
    return {
        "peak": float(irf[peak_t]),
        "peak_t": peak_t,
        "cumulative": float(irf.sum()),
        "half_life": half_life,
    }


def compute_metrics(irfs: dict[str, np.ndarray], variables: list[str]) -> dict[str, dict]:
    return {v: irf_metrics(irfs[v]) for v in variables}
