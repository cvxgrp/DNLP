import numpy as np

import cvxpy as cp

N = 1000
h = 1 / N
alpha = 350

t = cp.Variable(N+1, bounds=[-1, 1])
x = cp.Variable(N+1, bounds=[-0.05, 0.05])
u = cp.Variable(N+1)

control_terms = cp.multiply(0.5 * h, cp.power(u[1:], 2) + cp.power(u[:-1], 2))
trigonometric_terms = cp.multiply(0.5 * alpha * h, cp.cos(t[1:]) + cp.cos(t[:-1]))
objective_terms = cp.sum(control_terms + trigonometric_terms)

objective = cp.Minimize(objective_terms)

constraints = []

position_constraints = (x[1:] - x[:-1] - 
                       cp.multiply(0.5 * h, cp.sin(t[1:]) + cp.sin(t[:-1])) == 0)
constraints.append(position_constraints)

angle_constraint = (t[1:] - t[:-1] - 0.5 * h * (u[1:] + u[:-1]) == 0)
constraints.append(angle_constraint)

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True,
              derivative_test='none')
assert problem.status == cp.OPTIMAL
assert np.allclose(problem.value, 3.500e+02)
