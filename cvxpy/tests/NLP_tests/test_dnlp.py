import numpy as np
import pytest

import cvxpy as cp


class TestDNLP():
    """
    This class tests whether problems are correctly identified as DNLP
    (disciplined nonlinear programs) and whether the objective and constraints
    are correctly identified as smooth, esr or hsr.

    We adopt the convention that a function is smooth if and only if it is
    both esr and hsr. This convention is analogous to DCP and convex programming
    where a function is affine if and only if it is both convex and concave.
    """

    def test_simple_smooth(self):
        # Define a simple nonlinear program
        x = cp.Variable()
        y = cp.Variable()
        objective = cp.Minimize(cp.log(x - 1) + cp.exp(y - 2))
        constraints = [x + y == 1]
        prob = cp.Problem(objective, constraints)

        assert prob.is_dnlp()
        assert prob.objective.expr.is_smooth()
        assert prob.constraints[0].expr.is_smooth()

    def test_simple_esr(self):
        pass

    def test_simple_hsr(self):
        pass

    def test_simple_composition(self):
        pass

    def test_simple_non_dnlp(self):
        pass

    