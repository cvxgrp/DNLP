import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('PIPS' not in INSTALLED_SOLVERS, reason='PIPS is not installed.')
class TestExamplesPIPS:
    """
    Nonlinear test problems taken from the IPOPT documentation and
    the Julia documentation: https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/simple_examples/.
    """
    def test_hs071(self):
        x = cp.Variable(4, bounds=[0,6])
        x.value = np.array([1.0, 5.0, 5.0, 1.0])
        objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])

        constraints = [
            x[0]*x[1]*x[2]*x[3] >= 25,
            cp.sum(cp.square(x)) == 40,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.PIPS, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([0.75450865, 4.63936861, 3.78856881, 1.88513184]))
