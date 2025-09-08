import cvxpy as cp
import numpy as np

N = 4

# Conductance/susceptance components
G = np.array(
    [
        [1.7647, -0.5882, 0.0, -1.1765],
        [-0.5882, 1.5611, -0.3846, -0.5882],
        [0.0, -0.3846, 1.5611, -1.1765],
        [-1.1765, -0.5882, -1.1765, 2.9412],
    ]
)

B = np.array(
    [
        [-7.0588, 2.3529, 0.0, 4.7059],
        [2.3529, -6.629, 1.9231, 2.3529],
        [0.0, 1.9231, -6.629, 4.7059],
        [4.7059, 2.3529, 4.7059, -11.7647],
    ]
)

# Define bounds
v_lb = np.array([1.0, 0.0, 1.0, 0.0])
v_ub = np.array([1.0, 1.5, 1.0, 1.5])

P_lb = np.array([-3.0, -0.3, 0.3, -0.2])
P_ub = np.array([3.0, -0.3, 0.3, -0.2])

Q_lb = np.array([-3.0, -0.2, -3.0, -0.15])
Q_ub = np.array([3.0, -0.2, 3.0, -0.15])

theta_lb = np.array([0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])
theta_ub = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 2])

# Create variables with bounds included
P = cp.Variable(N, name="P", bounds=[P_lb, P_ub])
Q = cp.Variable(N, name="Q", bounds=[Q_lb, Q_ub])
v = cp.Variable(N, name="v", bounds=[v_lb, v_ub])
theta = cp.Variable(N, name="theta", bounds=[theta_lb, theta_ub])

# Reshape theta to column vector for broadcasting
theta_col = cp.reshape(theta, (N, 1))

# Create constraints list (only power balance constraints now)
constraints = []

# Real power balance
P_balance = cp.multiply(
    v,
    (
        G @ cp.cos(theta_col - theta_col.T)
        + B @ cp.sin(theta_col - theta_col.T)
    ) @ v
)
constraints.append(P == P_balance)

# Reactive power balance
Q_balance = cp.multiply(
    v,
    (
        G @ cp.sin(theta_col - theta_col.T)
        - B @ cp.cos(theta_col - theta_col.T)
    ) @ v
)
constraints.append(Q == Q_balance)

# Objective: minimize reactive power at buses 1 and 3 (indices 0 and 2)
objective = cp.Minimize(Q[0] + Q[2])

# Create and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, verbose=True, nlp=True)
