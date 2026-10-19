"""Impulse-response metrics.

Expect: signed peak, time of peak, cumulative sum and half-life (periods after the peak until |irf| <= half the peak).
"""

import numpy as np

from dgse_llm.experiments import compute_metrics, irf_metrics


def test_positive_hump():
    m = irf_metrics(np.array([0, 1, 2, 1, 0.5, 0.25, 0]))
    assert m["peak"] == 2 and m["peak_t"] == 2
    assert m["cumulative"] == 4.75
    assert m["half_life"] == 1


def test_negative_peak_is_signed():
    m = irf_metrics(np.array([0, -3.0, -1.0, 0]))
    assert m["peak"] == -3.0 and m["peak_t"] == 1 and m["half_life"] == 1


def test_never_halves_gives_none():
    assert irf_metrics(np.array([1.0, 1.0, 1.0]))["half_life"] is None


def test_zero_response():
    assert irf_metrics(np.zeros(5)) == {"peak": 0.0, "peak_t": 0, "cumulative": 0.0, "half_life": None}


def test_compute_metrics_keys():
    out = compute_metrics({"a": np.ones(3), "b": np.zeros(3)}, ["a"])
    assert list(out) == ["a"]
