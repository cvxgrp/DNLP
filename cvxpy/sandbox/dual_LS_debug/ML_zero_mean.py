from math import pi

import numpy as np

import cvxpy as cp
from cvxpy import log, square

np.random.seed(1234)
mu = cp.Variable((1, ), name="mu")
n = 2
data = np.random.randn(n)
sigma = cp.Variable((1, ), bounds =[0, 5])
obj = (n / 2) * log(2*pi*square(sigma)) + \
        (1 / (2 * square(sigma))) * cp.sum(cp.square(data-mu))
constraints = []
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True, print_level=12,
               hessian_approximation="exact",
               warm_start_init_point='no',  bound_mult_init_val=0.01)
