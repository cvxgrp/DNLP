import numpy as np

import cvxpy as cp

n = 100
Sigma = np.random.rand(n, n)
Sigma = Sigma @ Sigma.T  
mu = np.random.rand(n, )

x = cp.Variable((n, ), nonneg=True)

# This type of initialization makes ipopt muich more robust.
# With no initialization it sometimes fails. Perhaps this is 
# because we initialize in a very infeasible point?
x.value = np.ones(n) / n

obj = cp.square(mu @ x) / cp.quad_form(x, Sigma)
constraints = [cp.sum(x) == 1]
problem = cp.Problem(cp.Maximize(obj), constraints)
problem.solve(
	solver=cp.IPOPT, nlp=True, verbose=True, 
	hessian_approximation='exact', derivative_test='second-order'
)
x_noncvx = x.value
