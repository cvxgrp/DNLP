import numpy as np
import cvxpy as cp

p = np.array([.07, .12, .23, .19, .39])
x = cp.Variable(5)
prob = cp.Problem(cp.Maximize(cp.geo_mean(x, p)), [cp.sum(x) <= 1])
prob.solve(solver=cp.IPOPT, nlp=True, best_of=5)
x = np.array(x.value).flatten()
x_true = p/sum(p)

assert np.allclose(x, x_true, 1e-3)