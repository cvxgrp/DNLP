import cvxpy as cp
from cvxpy.reductions.expr2smooth.expr2smooth import Expr2Smooth


class TestSmithFormCanonicalization():
    """
    The following examples are taken from "GLOBAL OPTIMISATION OF
    GENERAL PROCESS MODELS" by Smith and Pantelides, Chapter 12 of
    some book.
    """
    def test_example1(self):
        """
        We consider the nonlinear expression:
        \frac{x * log(y) + z}{z + x*y}
        """
        x = cp.Variable((1,), name='x')
        y = cp.Variable((1,), name='y')
        z = cp.Variable((1,), name='z')

        obj = (x * cp.log(y) + z) / (z + x * y)
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)
        """
        We expect the canonicalized problem to be:
        min w6
        s.t w1 = log(y)
            w2 = x*w1
            w3 = w2 + z
            w4 = x*y
            w5 = z + w4
            w6 = w3/w5
        """
        assert str(new_prob.objective) == "minimize var20"
        assert len(new_prob.constraints) == 6
        assert str(new_prob.constraints[0]) == "var15 == log(y)"
        assert str(new_prob.constraints[1]) == "var16 == x @ var15"
        assert str(new_prob.constraints[2]) == "var17 == var16 + z"
        assert str(new_prob.constraints[3]) == "var18 == x @ y"
        assert str(new_prob.constraints[4]) == "var19 == z + var18"
        assert str(new_prob.constraints[5]) == "var20 == var17 / var19"

    def test_example2(self):
        """
        We consider the expression:
        \alpha * exp(\beta)(x + y)(x + gamma*y + delta*z)
        """
        alpha, beta, gamma, delta = 2.0, 3.0, 4.0, 5.0

        x = cp.Variable(name='x')
        y = cp.Variable(name='y')
        z = cp.Variable(name='z')

        obj = alpha * cp.exp(beta) * (x + y) * (x + gamma * y + delta * z)
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)
        """
        We expect the canonicalized problem to be:
        min w3
        s.t w1 = \alpha * exp(\beta)(x + y)
            w2 = x + gamma*y + delta*z
            w3 = w1 * w2
        """
        assert str(new_prob.objective) == "minimize var17"
        assert len(new_prob.constraints) == 3
        assert str(new_prob.constraints[0]) == "var15 == 2.0 @ exp(3.0) @ (x + y)"
        assert str(new_prob.constraints[1]) == "var16 == x + 4.0 @ y + 5.0 @ z"
        assert str(new_prob.constraints[2]) == "var17 == var15 @ var16"

    def test_example_div(self):
        """
        We consider the following expression:
        \frac{x + 2*y + z}{x + y}
        """
        x = cp.Variable((1,), name='x')
        y = cp.Variable((1,), name='y')
        z = cp.Variable((1,), name='z')

        obj = (x + 2*y + z) / (x + y)
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)
        """
        We expect the canonicalized problem to be:
        min w3
        s.t w1 = x + 2*y + z
            w2 = x + y
            w3 = w1/w2
        """
        assert str(new_prob.objective) == "minimize var11"
        assert len(new_prob.constraints) == 2
        assert str(new_prob.constraints[0]) == "var11 @ var12 == x + 2.0 @ y + z"
        assert str(new_prob.constraints[1]) == "var12 == x + y"

    def test_example_mul(self):
        """
        We consider the following expression:
        x(2y + z)
        """
        x = cp.Variable(name='x')
        y = cp.Variable(name='y')
        z = cp.Variable(name='z')

        obj = x * (2*y + z)
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)
        """
        We expect the canonicalized problem to be:
        min w2
        s.t w1 = 2*y + z
            w2 = x*w1
        """
        assert str(new_prob.objective) == "minimize var9"
        assert len(new_prob.constraints) == 2
        assert str(new_prob.constraints[0]) == "var8 == 2.0 @ y + z"
        assert str(new_prob.constraints[1]) == "var9 == x @ var8"
