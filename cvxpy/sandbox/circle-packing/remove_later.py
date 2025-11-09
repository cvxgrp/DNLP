import numpy as np

import cvxpy as cp

rng = np.random.default_rng(5)
n = 3
radius = rng.uniform(1.0, 3.0, n)

centers = cp.Variable((2, n), name='c')
constraints = []
for i in range(n - 1):
    for j in range(i + 1, n):
        constraints += [cp.sum(cp.square(centers[:, i] - centers[:, j])) >=
                        (radius[i] + radius[j]) ** 2]

centers.value = rng.uniform(-5.0, 5.0, (2, n))
obj = cp.Minimize(cp.max(cp.norm_inf(centers, axis=0) + radius))
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none',
            least_square_init_duals='no')

true_sol = np.array([[ 1.73655994, -1.98685738,  2.57208783],  
                        [ 1.99273311, -1.67415425, -2.57208783]])
assert np.allclose(centers.value, true_sol)