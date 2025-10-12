import numpy as np
import numpy.linalg as LA

import cvxpy as cp

# vanilla risk parity, small example with n = 8 assets

n = 8
Sigma = 1e-5 * np.array([[41.16, 22.03, 18.64, -4.74,  6.27, 10.1 , 14.52,  3.18],
                         [22.03, 58.57, 32.92, -5.04,  4.02,  3.7 , 26.76,  2.17],
                         [18.64, 32.92, 81.02,  0.53,  6.05,  2.02, 25.52,  1.56],
                         [-4.74, -5.04,  0.53, 20.6 ,  2.52,  0.57,  0.2 ,  3.6 ],
                         [6.27,  4.02,  6.05,  2.52, 10.13,  2.59,  4.32,  3.13],
                         [10.1 ,  3.7 ,  2.02,  0.57,  2.59, 22.89,  3.97,  3.26],
                         [14.52, 26.76, 25.52,  0.2 ,  4.32,  3.97, 29.91,  3.25],
                         [3.18,  2.17,  1.56,  3.6 ,  3.13,  3.26,  3.25, 13.63]])

risk_target = (1 / n) * np.ones(n)

w = cp.Variable((n, ), nonneg=True, name='w')
t = cp.Variable((n, ), name='t')
constraints = [cp.sum(w) == 1, t == Sigma @ w]

term1 = cp.sum(cp.multiply(cp.square(w), cp.square(t))) / cp.quad_form(w, Sigma)
term2 = (LA.norm(risk_target) ** 2) * cp.quad_form(w, Sigma)
term3 = - 2 * cp.sum(cp.multiply(risk_target, cp.multiply(w, t)))
obj = cp.Minimize(term1 + term2 + term3)
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='second-order')


risk_contributions = w.value * (Sigma @ w.value) 
risk_contributions /= np.sum(risk_contributions)
assert(np.linalg.norm(risk_contributions - risk_target) < 1e-5)