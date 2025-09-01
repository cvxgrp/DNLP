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
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()

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

    def test_example2(self):
        """
        We consider the expression:
        \alpha * exp(\beta)(x + y)(x + gamma*y + delta*z)
        """
        alpha, beta, gamma, delta = 2.0, 3.0, 4.0, 5.0
    
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()

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

    def test_example_div(self):
        """
        We consider the following expression:
        \frac{x + 2*y + z}{x + y}
        """
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()

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
    
    def test_example_mul(self):
        """
        We consider the following expression:
        x(2y + z)
        """
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()

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
