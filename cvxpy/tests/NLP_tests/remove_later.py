import cvxpy as cp
import numpy as np

# Variables (no global nonneg since x1 has lower bound 2)
x = cp.Variable(2, bounds=[np.array([2.0, 0.0]), np.array([np.inf, np.inf])])

objective = cp.Minimize(2 * x[0] + 3 * x[1])
constraints = [
    x[0] + x[1] <= 5,
]

prob = cp.Problem(objective, constraints)
prob.solve(nlp=True, solver=cp.PIPS)  # any LP-capable solver; default usually works

print("Status:", prob.status)
print("Objective:", prob.value)
print("x:", x.value)