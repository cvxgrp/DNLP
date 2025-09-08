#!/usr/bin/env python3
# CVXPY implementation of the bilinear optimization problem
# This example assumes CVXPY can handle non-convex bilinear constraints

import cvxpy as cp

# First version: continuous variables
print("=" * 50)
print("Version 1: Continuous variables")
print("=" * 50)

# Create variables (non-negative)
x = cp.Variable(nonneg=True, name="x")
y = cp.Variable(nonneg=True, name="y")
z = cp.Variable(nonneg=True, name="z")

# Set objective: maximize x
objective = cp.Maximize(x)

# Define constraints
constraints = [
    # Linear constraint: x + y + z <= 10
    x + y + z <= 10,
    
    # Bilinear inequality constraint: x * y <= 2
    x * y <= 2,
    
    # Bilinear equality constraint: x * z + y * z == 1
    x * z + y * z == 1
]

# Create and solve problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, verbose=True, nlp=True)

# Print results
print(f"Status: {problem.status}")
print(f"Optimal value: {problem.value:.6f}")
print(f"x = {x.value:.6f}")
print(f"y = {y.value:.6f}")
print(f"z = {z.value:.6f}")
