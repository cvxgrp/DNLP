import numpy as np
from scipy.linalg import eigh
import cvxpy as cp

# Generate real data for a structural vibration problem
np.random.seed(42)
n = 10  # dimension
K_random = np.random.randn(n, n)
K = K_random @ K_random.T + 5 * np.eye(n)
M_random = np.random.randn(n, n) * 0.3
M = M_random @ M_random.T + np.eye(n)

print("Stiffness matrix (K) - first 5x5 block:")
print(K[:5, :5].round(2))
print("\nMass matrix (M) - first 5x5 block:")
print(M[:5, :5].round(2))

# Direct non-convex Rayleigh quotient formulation
def rayleigh_quotient_direct(K, M, find_min=True):
    x = cp.Variable(n)
    numerator = cp.quad_form(x, K)
    denominator = cp.quad_form(x, M)
    rayleigh_quotient = numerator / denominator
    objective = cp.Minimize(rayleigh_quotient) if find_min else cp.Maximize(rayleigh_quotient)
    constraints = [cp.norm(x, 2) == 1]
    problem = cp.Problem(objective, constraints)
    print(f"\n{'='*50}")
    print(f"Solving for {'minimum' if find_min else 'maximum'} Rayleigh quotient")
    x_init = np.random.randn(n)
    x_init = x_init / np.linalg.norm(x_init)
    x.value = x_init
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
    if x.value is not None:
        rq_value = (x.value.T @ K @ x.value) / (x.value.T @ M @ x.value)
        return x.value, rq_value
    else:
        print("Solver failed to find a solution")
        return None, None

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

x_min, rq_min = rayleigh_quotient_direct(K, M, find_min=True)
if x_min is not None:
    print(f"\nMinimum Rayleigh quotient found: {rq_min:.6f}")
    print(f"Ground truth minimum: {min_eigenvalue:.6f}")
    print(f"Error: {abs(rq_min - min_eigenvalue):.6e}")

x_max, rq_max = rayleigh_quotient_direct(K, M, find_min=False)
if x_max is not None:
    print(f"\nMaximum Rayleigh quotient found: {rq_max:.6f}")
    print(f"Ground truth maximum: {max_eigenvalue:.6f}")
    print(f"Error: {abs(rq_max - max_eigenvalue):.6e}")
