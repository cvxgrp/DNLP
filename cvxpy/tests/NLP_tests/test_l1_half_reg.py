import numpy as np

import cvxpy as cp

# Generate random data
np.random.seed(0)
m, n = 40, 40
A = np.random.randn(m, n)
b = np.random.randn(m)
# Define the variable
x = cp.Variable(n)
# set initial value for x
#xls = np.array([ 0.13023767,  0.09473619,  0.20023978,  0.129647  , -0.26661262,
 #      -0.18313258,  0.29880278,  0.10479523, -0.14954388,  0.32831736])
xls = np.linalg.lstsq(A, b, rcond=None)[0]
x.value = np.ones(n)
gamma = 1
# Define the objective function with l1-2 norm
regularizer = 0
for i in range(n):
    regularizer += gamma*cp.abs(x[i])
objective = cp.Minimize(cp.sum_squares(A @ x - b) + regularizer)
problem = cp.Problem(objective)
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True, derivative_test='none', verbose=True)
print(x.value[np.abs(x.value) < 1e-3])
print("Optimal value:", problem.value)
#print(np.linalg.norm(A @ xls - b)**2 + gamma * np.sum(np.abs(xls)))
#problem.solve(solver=cp.CLARABEL)
#print("Optimal value with clarabel:", problem.value)
print(x.value)