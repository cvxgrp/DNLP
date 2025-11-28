import cvxpy as cp
import numpy as np

np.random.seed(858)
n = 100
x = cp.Variable(n, name='x')
mu = np.random.randn(n)
Sigma = np.random.randn(n, n)
Sigma = Sigma.T @ Sigma
gamma = 0.1
t = cp.Variable(name='t', bounds=[0, None])
L = np.linalg.cholesky(Sigma, upper=False)

objective = cp.Minimize(- mu.T @ x + gamma * t)
constraints = [cp.norm(L.T @ x, 2) <= t,
                cp.sum(x) == 1,
                x >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.KNITRO, nlp=True)
assert problem.status == cp.OPTIMAL
assert np.allclose(problem.value, -1.93414338e+00)