import numpy as np

import cvxpy as cp

# Generate random data
np.random.seed(0)
m, n = 20, 20
A = np.random.randn(m, 2*n)
B = np.random.randn(m, n)
b = np.random.randn(m)
# Define the variable
x1 = cp.Variable(2*n)
x2 = cp.Variable(n)
gamma = 20
# Define the objective function with l1-2 norm
regularizer = gamma * (cp.norm2(x1) + cp.norm2(x2))
objective = cp.Minimize(cp.sum_squares(A @ x1 + B @ x2 - b) + regularizer)
problem = cp.Problem(objective)
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True, derivative_test='none', verbose=True)
#print(len(x1.value[np.abs(x1.value) < 1e-3]))
#print(len(x2.value[np.abs(x2.value) < 1e-3]))
print("Optimal value:", problem.value)
#problem.solve(solver=cp.CLARABEL)
print("Optimal value with clarabel:", problem.value)
print(x1.value)
print(x2.value)