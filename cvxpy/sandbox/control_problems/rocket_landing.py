import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp

# Data - all parameters normalized to be dimensionless
h_0 = 1                      # Initial height
v_0 = 0                      # Initial velocity
m_0 = 1.0                    # Initial mass
m_T = 0.6                    # Final mass
g_0 = 1                      # Gravity at the surface
h_c = 500                    # Used for drag
c = 0.5 * np.sqrt(g_0 * h_0) # Thrust-to-fuel mass
D_c = 0.5 * 620 * m_0 / g_0  # Drag scaling
u_t_max = 3.5 * g_0 * m_0    # Maximum thrust
T_max = 0.2                  # Number of seconds
T = 2                        # Number of time steps
dt = 0.2 / T                 # Time per discretized step

# Create variables
x_v = cp.Variable(T, bounds=[0, np.inf], name="velocity")  # Velocity
x_h = cp.Variable(T, bounds=[0, np.inf], name="height")  # Height
x_m = cp.Variable(T, name="mass")                      # Mass
u_t = cp.Variable(T, bounds=[0, u_t_max], name="thrust") # Thrust

# Set starting values (equivalent to JuMP's start parameter)
x_v.value = np.full(T, v_0)        # start = v_0
x_h.value = np.full(T, h_0)        # start = h_0
x_m.value = np.full(T, m_0)        # start = m_0
u_t.value = np.zeros(T)            # start = 0

# Initialize constraints list
constraints = []

# Boundary conditions
constraints.append(x_v[0] == v_0)
constraints.append(x_h[0] == h_0)
constraints.append(x_m[0] == m_0)
constraints.append(u_t[T-1] == 0.0)

# Mass constraints
constraints.append(x_m >= m_T)

# Thrust constraints
constraints.append(u_t <= u_t_max)

# Create slices for t=1:T and t=0:T-1
# These are views, not copies
x_h_curr = x_h[1:T]      # x_h[t] for t = 1, 2, ..., T-1
x_h_prev = x_h[0:T-1]    # x_h[t-1] for t = 1, 2, ..., T-1

x_v_curr = x_v[1:T]
x_v_prev = x_v[0:T-1]

x_m_curr = x_m[1:T]
x_m_prev = x_m[0:T-1]

u_t_prev = u_t[0:T-1]    # u_t[t-1] for t = 1, 2, ..., T-1

# Vectorized constraint 1: Rate of ascent
# (x_h[t] - x_h[t-1])/dt = x_v[t-1] for all t = 1, ..., T-1
constraints.append((x_h_curr - x_h_prev) / dt == x_v_prev)

# Vectorized constraint 2: Acceleration
# Compute drag force for all timesteps at once
drag_force = D_c * cp.square(x_v_prev) * cp.exp(-h_c * (x_h_prev - h_0) / h_0)

# Compute gravity force for all timesteps at once
gravity_force = g_0 * (cp.square(h_0) / cp.square(x_h_prev))

# Apply the acceleration constraint for all timesteps
constraints.append(
    (x_v_curr - x_v_prev) / dt == (u_t_prev - drag_force) / x_m_prev - gravity_force
)

# Vectorized constraint 3: Rate of mass loss
# (x_m[t] - x_m[t-1])/dt = -u_t[t-1]/c for all t = 1, ..., T-1
constraints.append((x_m_curr - x_m_prev) / dt == -u_t_prev / c)

# Objective: maximize altitude at end of time of flight
objective = cp.Maximize(x_h[T-1])

# Create and solve problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# Check if solution was found
if problem.status == cp.OPTIMAL:
    print(f"Optimal value: {problem.value}")
    print(f"Final altitude: {x_h.value[T-1]}")
    print(f"Final mass: {x_m.value[T-1]}")
    print(f"Final velocity: {x_v.value[T-1]}")
else:
    print(f"Problem status: {problem.status}")

# Plot results if solution found
if problem.status == cp.OPTIMAL:
    # Create time array
    time = np.arange(T) * dt
    
    # Create figure with subplots
    plt.figure(figsize=(12, 8))
    
    # Plot altitude
    plt.subplot(2, 2, 1)
    plt.plot(time, x_h.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude')
    plt.grid(True)
    
    # Plot mass
    plt.subplot(2, 2, 2)
    plt.plot(time, x_m.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Mass')
    plt.grid(True)
    
    # Plot velocity
    plt.subplot(2, 2, 3)
    plt.plot(time, x_v.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity')
    plt.grid(True)
    
    # Plot thrust
    plt.subplot(2, 2, 4)
    plt.plot(time, u_t.value)
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nSolution Statistics:")
    print(f"Maximum altitude reached: {np.max(x_h.value):.6f}")
    print(f"Maximum velocity reached: {np.max(x_v.value):.6f}")
    print(f"Maximum thrust used: {np.max(u_t.value):.6f}")
    print(f"Total fuel consumed: {m_0 - x_m.value[T-1]:.6f}")
else:
    print("No optimal solution found. Check problem formulation.")
