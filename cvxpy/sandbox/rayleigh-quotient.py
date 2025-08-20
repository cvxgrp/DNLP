import numpy as np
from scipy.linalg import eigh

import cvxpy as cp

# Generate real data for a structural vibration problem
# A represents stiffness matrix, B represents mass matrix
np.random.seed(42)
n = 10  # dimension

# Create symmetric positive definite matrices
# Stiffness matrix (typically has larger eigenvalues)
K_random = np.random.randn(n, n)
K = K_random @ K_random.T + 5 * np.eye(n)  # Stiffness matrix

# Mass matrix (typically has eigenvalues around 1)
M_random = np.random.randn(n, n) * 0.3
M = M_random @ M_random.T + np.eye(n)  # Mass matrix

print("Stiffness matrix (K) - first 5x5 block:")
print(K[:5, :5].round(2))
print("\nMass matrix (M) - first 5x5 block:")
print(M[:5, :5].round(2))

# Direct non-convex Rayleigh quotient formulation
def rayleigh_quotient_direct(K, M, find_min=True):
    """
    Direct formulation of Rayleigh quotient problem:
    
    minimize/maximize: (x^T K x) / (x^T M x)
    subject to: ||x||_2 = 1
    
    This is the pure, non-convex formulation.
    """
    # Decision variable
    x = cp.Variable(n)
    
    # Rayleigh quotient: (x^T K x) / (x^T M x)
    numerator = cp.quad_form(x, K)
    denominator = cp.quad_form(x, M)
    rayleigh_quotient = numerator / denominator
    
    # Objective
    if find_min:
        objective = cp.Minimize(rayleigh_quotient)
    else:
        objective = cp.Maximize(rayleigh_quotient)
    
    # Constraint: unit norm to avoid trivial solution
    constraints = [
        cp.norm(x, 2) == 1
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    print(f"\n{'='*50}")
    print(f"Solving for {'minimum' if find_min else 'maximum'} Rayleigh quotient")
    
    # Solve with a non-convex solver
    # SCS can sometimes handle non-convex problems
    try:
        # Set initial guess
        x_init = np.random.randn(n)
        x_init = x_init / np.linalg.norm(x_init)
        x.value = x_init
        
        # Solve
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        
        if x.value is not None:
            # Compute the Rayleigh quotient value
            rq_value = (x.value.T @ K @ x.value) / (x.value.T @ M @ x.value)
            return x.value, rq_value
        else:
            print("Solver failed to find a solution")
            return None, None
            
    except Exception as e:
        print(f"Error during solving: {e}")
        return None, None
"""
# Alternative formulation: minimize quadratic form with quadratic constraint
def rayleigh_quotient_alternative(K, M, find_min=True):
    # Decision variable
    x = cp.Variable(n)
    
    # Objective: quadratic form with K
    if find_min:
        objective = cp.Minimize(cp.quad_form(x, K))
    else:
        objective = cp.Maximize(cp.quad_form(x, K))
    
    # Constraint: quadratic form with M equals 1
    constraints = [
        cp.quad_form(x, M) == 1
    ]
    
    # Create problem
    problem = cp.Problem(objective, constraints)
    
    print(f"\n{'='*50}")
    print(f"Alternative formulation - {'minimum' if find_min else 'maximum'}")
    
    try:
        # Set initial guess
        x_init = np.random.randn(n)
        x_init = x_init / np.sqrt(x_init.T @ M @ x_init)
        x.value = x_init
        
        # Solve
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        
        if x.value is not None:
            # The objective value is the Rayleigh quotient since x^T M x = 1
            rq_value = x.value.T @ K @ x.value
            return x.value, rq_value
        else:
            print("Solver failed to find a solution")
            return None, None
            
    except Exception as e:
        print(f"Error during solving: {e}")
        return None, None
"""
# Ground truth using scipy
print("="*50)
print("Ground Truth (using scipy.linalg.eigh):")
eigenvalues, eigenvectors = eigh(K, M)
min_eigenvalue = eigenvalues[0]
min_eigenvector = eigenvectors[:, 0]
max_eigenvalue = eigenvalues[-1]
max_eigenvector = eigenvectors[:, -1]

print(f"Minimum eigenvalue: {min_eigenvalue:.6f}")
print(f"Maximum eigenvalue: {max_eigenvalue:.6f}")
print(f"Eigenvalue range: [{min_eigenvalue:.6f}, {max_eigenvalue:.6f}]")

# Solve using direct formulation
print("\n" + "="*50)
print("DIRECT NON-CONVEX FORMULATION")

# Find minimum
x_min, rq_min = rayleigh_quotient_direct(K, M, find_min=True)
if x_min is not None:
    print(f"\nMinimum Rayleigh quotient found: {rq_min:.6f}")
    print(f"Ground truth minimum: {min_eigenvalue:.6f}")
    print(f"Error: {abs(rq_min - min_eigenvalue):.6e}")

# Find maximum
x_max, rq_max = rayleigh_quotient_direct(K, M, find_min=False)
if x_max is not None:
    print(f"\nMaximum Rayleigh quotient found: {rq_max:.6f}")
    print(f"Ground truth maximum: {max_eigenvalue:.6f}")
    print(f"Error: {abs(rq_max - max_eigenvalue):.6e}")

# Solve using alternative formulation
print("\n" + "="*50)
print("ALTERNATIVE NON-CONVEX FORMULATION")

"""
# Find minimum
x_min_alt, rq_min_alt = rayleigh_quotient_alternative(K, M, find_min=True)
if x_min_alt is not None:
    print(f"\nMinimum Rayleigh quotient found: {rq_min_alt:.6f}")
    print(f"Ground truth minimum: {min_eigenvalue:.6f}")
    print(f"Error: {abs(rq_min_alt - min_eigenvalue):.6e}")

# Find maximum  
x_max_alt, rq_max_alt = rayleigh_quotient_alternative(K, M, find_min=False)
if x_max_alt is not None:
    print(f"\nMaximum Rayleigh quotient found: {rq_max_alt:.6f}")
    print(f"Ground truth maximum: {max_eigenvalue:.6f}")
    print(f"Error: {abs(rq_max_alt - max_eigenvalue):.6e}")

# Verification
print("\n" + "="*50)
print("VERIFICATION OF SOLUTIONS")
if x_min is not None:
    print("\nDirect formulation (minimum):")
    print(f"  ||x||_2 = {np.linalg.norm(x_min):.6f}")
    print(f"  x^T K x = {x_min.T @ K @ x_min:.6f}")
    print(f"  x^T M x = {x_min.T @ M @ x_min:.6f}")
    print(f"  Rayleigh quotient = {(x_min.T @ K @ x_min) / (x_min.T @ M @ x_min):.6f}")

if x_min_alt is not None:
    print("\nAlternative formulation (minimum):")
    print(f"  ||x||_2 = {np.linalg.norm(x_min_alt):.6f}")
    print(f"  x^T K x = {x_min_alt.T @ K @ x_min_alt:.6f}")
    print(f"  x^T M x = {x_min_alt.T @ M @ x_min_alt:.6f}")
    print(f"  Rayleigh quotient = {(x_min_alt.T @ K @ x_min_alt) / (x_min_alt.T @ M @ x_min_alt):.6f}")
"""