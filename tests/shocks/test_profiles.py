"""Time profiles.

Expect: unit-height shapes of length T; one_time is a single impulse, multi_period holds for `duration`,
geometric_decay is rho**k from start, from_series is placed at start and truncated at the horizon.
"""

import numpy as np
import pytest

from dgse_llm.schemas import TimeProfile
from dgse_llm.shocks import PROFILE_REGISTRY, make_shape

T = 10


def shape(**kw):
    return make_shape(T, TimeProfile(**kw))


def test_one_time():
    s = shape(kind="one_time", start=3)
    assert s.shape == (T,) and s[3] == 1 and s.sum() == 1


def test_multi_period():
    s = shape(kind="multi_period", start=2, duration=4)
    assert np.array_equal(s, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])


def test_multi_period_truncated_at_horizon():
    assert shape(kind="multi_period", start=8, duration=5).sum() == 2


def test_geometric_decay():
    s = shape(kind="geometric_decay", start=2, rho=0.5)
    assert np.allclose(s[2:5], [1, 0.5, 0.25]) and s[:2].sum() == 0


def test_from_series():
    s = shape(kind="from_series", start=1, series=[3.0, 2.0, 1.0])
    assert np.array_equal(s[:5], [0, 3, 2, 1, 0])
    assert shape(kind="from_series", start=8, series=[1.0] * 5).sum() == 2


def test_start_beyond_horizon_is_all_zero():
    assert shape(kind="one_time", start=50).sum() == 0
    assert shape(kind="geometric_decay", start=50).sum() == 0


def test_narrative_timeline_has_no_fixed_shape():
    with pytest.raises(ValueError):
        shape(kind="narrative_timeline")


def test_every_schema_kind_is_registered():
    assert {"one_time", "multi_period", "geometric_decay", "from_series", "at_periods"} <= set(PROFILE_REGISTRY.names())


def test_at_periods_single_and_several():
    assert np.array_equal(shape(kind="at_periods", periods=[4]), [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert np.array_equal(shape(kind="at_periods", periods=[1, 5, 8]), [0, 1, 0, 0, 0, 1, 0, 0, 1, 0])


def test_at_periods_ignores_periods_beyond_the_horizon():
    assert shape(kind="at_periods", periods=[2, 50]).sum() == 1
