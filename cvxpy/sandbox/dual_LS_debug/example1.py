
import numpy as np

import cvxpy as cp

x = cp.Variable((1, ), name="x", nonneg=True)
y = cp.Variable((1, ), name="y", nonneg=True)
z = cp.Variable((1, ), name="z", nonneg=True)

x = cp.Variable((1, ), name="x", bounds=[-1, 1])
y = cp.Variable((1, ), name="y", bounds=[None, 1])
z = cp.Variable((1, ), name="z", bounds=[None, None])

x.value = np.array([0.0])
y.value = np.array([0.0])
z.value = np.array([0.0])
constraints =[x + 2 * y == 3, x + z == 4]


obj = cp.square(x) - 6 * x + cp.square(y) - 3 * y + cp.square(z) - 4 * z

prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(solver=cp.IPOPT, nlp=True, verbose=True, hessian_approximation="exact", max_iter=0)