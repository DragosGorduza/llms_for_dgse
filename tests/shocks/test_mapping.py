"""Mapping functions.

Expect: identity, linear and pct give the documented values; unknown names and bad parameters raise.
"""

import pytest

from dgse_llm.shocks import MAPPING_REGISTRY, apply_mapping


def test_identity():
    assert apply_mapping("identity", 2.5, {}) == 2.5


def test_linear():
    assert apply_mapping("linear", -40.0, {"slope": -0.001}) == pytest.approx(0.04)
    assert apply_mapping("linear", 2.0, {"slope": 3.0, "intercept": 1.0}) == 7.0


def test_pct_matches_notebook_function():
    assert apply_mapping("pct", -20.0, {"factor": 0.5}) == pytest.approx(-0.1)


def test_unknown_mapping_lists_available():
    with pytest.raises(KeyError, match="Available"):
        apply_mapping("nope", 1.0, {})


def test_bad_parameter_raises():
    with pytest.raises(TypeError):
        apply_mapping("identity", 1.0, {"slope": 1.0})


def test_registry_contents():
    assert {"identity", "linear", "pct"} <= set(MAPPING_REGISTRY.names())
