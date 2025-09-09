import cvxpy as cp
import numpy as np
from cvxpy.reductions.expr2smooth.expr2smooth import Expr2Smooth
from cvxpy.lin_ops.lin_utils import ID_COUNTER

class TestSmithFormCanonicalization():
   
    def test_NMF_mul_canon_1(self):
        np.random.seed(858)
        ID_COUNTER.count = 1
        n, m, k = 4, 4, 4
        noise_level = 0.05
        X_true = np.random.rand(n, k)
        Y_true = np.random.rand(k, m)
        A_noisy = X_true @ Y_true 
        A_noisy = np.clip(A_noisy, 0, None)
        X = cp.Variable((n, k), bounds=[0, None], name='X')
        Y = cp.Variable((k, m), bounds=[0, None], name='Y')

        obj = cp.sum(cp.square(A_noisy - X @ Y))
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)

        assert str(new_prob.objective) == "minimize Sum(power(var12, 2.0), None, False)"
        assert len(new_prob.constraints) == 1
        assert str(new_prob.constraints[0]) == "var12 == [[0.60 0.44 0.66 0.79]\n [1.24 1.04 1.01 1.19]\n [1.09 0.86 1.04 1.13]\n [0.38 0.23 0.35 0.83]] + -X @ Y"

    def test_NMF_mul_canon_2(self):
        np.random.seed(858)
        ID_COUNTER.count = 1
        n, m, k = 4, 4, 4
        noise_level = 0.05
        X_true = np.random.rand(n, k)
        Y_true = np.random.rand(k, m)
        A_noisy = X_true @ Y_true 
        A_noisy = np.clip(A_noisy, 0, None)
        X = cp.Variable((n, k), bounds=[0, None], name='X')
        Y = cp.Variable((k, m), bounds=[0, None], name='Y')
        
        obj = cp.sum(cp.square(A_noisy - (X @ X) @ Y))
        problem = cp.Problem(cp.Minimize(obj), [])
        reduction = Expr2Smooth(problem)
        new_prob, inv_data = reduction.apply(problem)

        assert str(new_prob.objective) == "minimize Sum(power(var19, 2.0), None, False)"
        assert len(new_prob.constraints) == 2
        assert str(new_prob.constraints[0]) == "var10 == X @ X"
        assert str(new_prob.constraints[1]) == "var19 == [[0.60 0.44 0.66 0.79]\n [1.24 1.04 1.01 1.19]\n [1.09 0.86 1.04 1.13]\n [0.38 0.23 0.35 0.83]] + -var10 @ Y"


