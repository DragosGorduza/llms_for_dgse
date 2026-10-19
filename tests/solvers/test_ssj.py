"""SSJ adapter (placeholder).

Expect: models without a to_ssj() hook are refused with a pointer to the numpy solver.
"""

import pytest

from dgse_llm.solvers import build_solver


def test_model_without_hook_is_refused(models):
    m = models["rbc"]
    with pytest.raises(NotImplementedError, match="to_ssj"):
        build_solver("ssj").solve(m, m.baseline_exog(10), 10)
