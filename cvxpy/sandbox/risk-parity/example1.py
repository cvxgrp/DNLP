import numpy as np
import numpy.linalg as LA

import cvxpy as cp

# vanilla risk parity, small example with n = 3 assets

Sigma = np.array([[1.0000, 0.0015, -0.0119],
                  [0.0015, 1.0000, -0.0308],
                   [-0.0119, -0.0308, 1.0000]])

b = np.array([0.1594, 0.0126, 0.8280])
n = 3

w = cp.Variable((n, ), nonneg=True, name='w')
t = cp.Variable((n, ), name='t')
w.value = np.array([1/3, 1/3, 1/3])
constraints = [cp.sum(w) == 1, t == Sigma @ w]

term1 = cp.sum(cp.multiply(cp.square(w), cp.square(t))) / cp.quad_form(w, Sigma)
term2 = (LA.norm(b) ** 2) * cp.quad_form(w, Sigma)
term3 = - 2 * cp.sum(cp.multiply(b, cp.multiply(w, t)))
obj = cp.Minimize(term1 + term2 + term3)
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='second-order')


risk_contributions = w.value * (Sigma @ w.value) 
risk_contributions /= np.sum(risk_contributions)
print("risk contributions:         ", risk_contributions)
print("budgeted risk contributions:", b)