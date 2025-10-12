import numpy as np
import numpy.linalg as LA

import cvxpy as cp

np.random.seed(0)
m, n = 40, 40
factors = np.linspace(0.1, 1, 20)

for factor in factors:
    b = np.random.randn(m)
    A = np.random.randn(m, n)
    lmbda_max = 2 * LA.norm(A.T @ b, np.inf)
    lmbda = factor * lmbda_max

    x = cp.Variable((n, ), name='x', bounds = [-5, 5])
    obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve(solver=cp.CLARABEL)
    obj_star_dcp = obj.value

    x = cp.Variable((n, ), name='x')
    obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
    problem = cp.Problem(cp.Minimize(obj))
    problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact',
                    derivative_test='none', verbose=True, warm_start_init_point='no',
                     mu_strategy='adaptive',
                    max_iter=0)
    obj_star_nlp = obj.value
    import pdb
    pdb.set_trace()
    assert(np.abs(obj_star_nlp - obj_star_dcp) / obj_star_nlp <= 1e-4)

    