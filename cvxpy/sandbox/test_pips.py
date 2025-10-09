#!/usr/bin/env python
"""
Simple example of using PIPS for a linear programming problem
"""

import numpy as np
from numpy import array, dot, float64, r_  # symbols for nonlinear example
from pypower.pips import pips
from scipy.sparse import csr_matrix


# Define the linear objective function
def linear_objective(x, return_hessian=False):
    """
    Minimize: f(x) = 2*x1 + 3*x2
    """
    # Objective value
    f = 2*x[0] + 3*x[1]
    
    # Gradient (constant for linear problems)
    df = np.array([2.0, 3.0])
    
    if return_hessian:
        # Hessian (zero for linear problems)
        d2f = np.zeros((2, 2))
        return f, df, d2f
    else:
        return f, df

# Starting point
x0 = np.array([1.0, 1.0])

# Linear constraint: x1 + x2 <= 5
A = csr_matrix([[1, 1]])
lin_lower = np.array([-np.inf])  # no lower bound
lin_upper = np.array([5.0])      # upper bound of 5

# Variable bounds: x1 >= 0, x2 >= 0
xmin = np.array([2.0, 0.0])
xmax = np.array([np.inf, np.inf])

# Solve
print("Solving: minimize 2*x1 + 3*x2")
print("Subject to: x1 + x2 <= 5, x1 >= 0, x2 >= 0")
print("-" * 40)

solution = pips(linear_objective, x0, A, lin_lower, lin_upper, xmin, xmax)

print(f"Solution: x = {solution['x']}")
print(f"Objective value: {solution['f']:.4f}")
print(f"Converged: {solution['eflag']}")


# -------------------------------------------------------------
# Nonlinear example from PIPS docstring (Wikipedia NLP example)
# -------------------------------------------------------------


def f2(x):
    """Nonlinear objective used in Wikipedia example.

    f(x) = -x0*x1 - x1*x2
    Since we provide an explicit Hessian callback (`hess2`), PIPS only
    expects this function to return (f, grad).
    """
    f = -x[0] * x[1] - x[1] * x[2]
    df = -r_[x[1], x[0] + x[2], x[1]]
    return f, df


def gh2(x):
    """Nonlinear inequality constraints h(x) <= 0; no equalities.

    h(x) = [  x0^2 - x1^2 + x2^2 - 2,
              x0^2 + x1^2 + x2^2 - 10 ]
    """
    h = dot(array([[1, -1, 1],
                   [1,  1, 1]]), x**2) + array([-2.0, -10.0])
    dh = 2 * csr_matrix(array([[ x[0], x[0]],
                               [-x[1], x[1]],
                               [ x[2], x[2]]]))
    g = array([])  # no equalities
    dg = None
    return h, g, dh, dg


def hess2(x, lam, cost_mult=1):
    """Hessian of Lagrangian for the nonlinear example.

    lam: dict with key 'ineqnonlin' giving multipliers for h(x).
    """
    mu = lam["ineqnonlin"]
    a = r_[dot(2 * array([1, 1]), mu), -1, 0]
    b = r_[-1, dot(2 * array([-1, 1]), mu), -1]
    c = r_[0, -1, dot(2 * array([1, 1]), mu)]
    Lxx = csr_matrix(array([a, b, c]))
    return Lxx


print("\nSolving nonlinear example from PIPS docstring ...")
x0_nl = array([1, 1, 0], float64)
solution_nl = pips(f2, x0_nl, gh_fcn=gh2, hess_fcn=hess2)
print(f"Nonlinear solution: x = {solution_nl['x']}")
print(f"Objective value: {solution_nl['f']}")
print(f"Iterations: {solution_nl['output']['iterations']}")
print(f"Check objective ~ -7.07106725919: {round(solution_nl['f'], 11) == -7.07106725919}")
