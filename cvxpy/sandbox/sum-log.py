# write a log sum problem in cvxpy
import numpy as np

import cvxpy as cp

# Generate random data
np.random.seed(0)
m, n = 1, 2
A = np.random.randn(m, n)
b = np.random.randn(m)
# Define the variable
x = cp.Variable(n)
# set initial value for x
objective = cp.Minimize(-cp.sum(cp.log(A @ x - b)))
problem = cp.Problem(objective, [])
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True)
print("Optimal value:", problem.value)
