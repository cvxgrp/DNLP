import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS

np.random.seed(1)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestPricingProblem:
    n = 1000
    m = 30
    E = -0.1 + 0.2 * np.random.rand(n, n)
    np.fill_diagonal(E, -2 + 1.5 * np.random.rand(n))
    rnom = 1 + 4 * np.random.rand(n)
    knom = 0.9 * rnom
    A = np.random.randn(n, m)
    lower, upper = np.log(0.8), np.log(1.2)

    # problem
    pi = cp.Variable(n, bounds=[lower, upper])
    delta = cp.Variable(n)
    theta = cp.Variable(m)

    pi.value = np.zeros(n)
    delta.value = np.zeros(n)
    theta.value = np.zeros(m)
    obj = cp.sum(cp.multiply(rnom, cp.exp(delta + pi))) - cp.sum(cp.multiply(knom, cp.exp(delta)))

    print(obj.value)
    constr = [delta == E @ pi, pi == A @ theta]
    prob = cp.Problem(cp.Maximize(obj), constr)
    prob.solve(solver=cp.IPOPT, nlp=True, derivative_test='none', verbose=True)
