import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def solve_car_control_vectorized(x_final, L=0.1, N=50, h=0.1, gamma=10,
                                speed_bounds=(-0.6, 0.8), 
                                steering_bounds=(-np.pi/8, np.pi/6),
                                accel_bound=0.4, 
                                steering_rate_bound=np.pi/10):
    """
    Solve the nonlinear optimal control problem for car trajectory planning.
    
    Parameters:
    - x_final: tuple (p1, p2, theta) for final position and orientation
    - L: wheelbase length
    - N: number of time steps
    - h: time step size
    - gamma: weight for control smoothness term
    - speed_bounds: tuple (min_speed, max_speed) in m/s
    - steering_bounds: tuple (min_steering, max_steering) in radians
    - accel_bound: maximum acceleration magnitude in m/s²
    - steering_rate_bound: maximum steering rate magnitude in rad/s
    
    Returns:
    - x_opt: optimal states (N+1 x 3)
    - u_opt: optimal controls (N x 2)
    """
    # Add random seed for reproducibility
    np.random.seed(858)
    x, u = cp.Variable((N+1, 3)), cp.Variable((N, 2))
    u.value = np.random.uniform(0, 1, size=(N,2))
    x_init = np.array([0, 0, 0])

    objective = cp.sum_squares(u)
    objective += gamma * cp.sum_squares(u[1:, :] - u[:-1, :])

    constraints = [x[0, :] == x_init, x[N, :] == x_final]
    # Extract state components for timesteps 0 to N-1
    x_curr, x_next = x[:-1, :], x[1:, :]
    v, delta, theta = u[:, 0], u[:, 1], x_curr[:, 2]

    constraints += [x_next[:, 0] == x_curr[:, 0] + h * cp.multiply(v, cp.cos(theta)),
                    x_next[:, 1] == x_curr[:, 1] + h * cp.multiply(v, cp.sin(theta)),
                    x_next[:, 2] == x_curr[:, 2] + h * cp.multiply(v / L, cp.tan(delta))]

    # speed limit bounds (parameterized)
    constraints += [u[:, 0] >= speed_bounds[0], u[:, 0] <= speed_bounds[1]]
    # steering angle bounds (parameterized)
    constraints += [u[:, 1] >= steering_bounds[0], u[:, 1] <= steering_bounds[1]]
    # acceleration bounds (parameterized)
    constraints += [cp.abs(u[1:, 0] - u[:-1, 0]) <= accel_bound * h]
    # steering angle rate bounds (parameterized)
    constraints += [cp.abs(u[1:, 1] - u[:-1, 1]) <= steering_rate_bound * h]
    
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
    return x.value, u.value

def plot_trajectory(x_opt, u_opt, L):
    """
    Plot the car trajectory with orientation indicators.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    car_length = L
    car_width = L * 0.6

    # Select time steps to show car outline (e.g., every 2nd step for more shadows)
    steps_to_show = np.arange(0, len(x_opt), max(1, len(x_opt)//20))
    n_shadows = len(steps_to_show)

    # Draw car as a fading rectangle (shadow) at each step
    for i, k in enumerate(steps_to_show):
        p1, p2, theta = x_opt[k]
        # Rectangle corners (centered at (p1, p2), rotated by theta)
        corners = np.array([
            [ car_length/2,  car_width/2],
            [ car_length/2, -car_width/2],
            [-car_length/2, -car_width/2],
            [-car_length/2,  car_width/2],
            [ car_length/2,  car_width/2],  # close the rectangle
        ])
        # Rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = (R @ corners.T).T + np.array([p1, p2])
        # Fade older shadows
        alpha = 0.15 + 0.7 * (i+1)/n_shadows
        ax.fill(rotated[:,0], rotated[:,1], color='dodgerblue', alpha=alpha, edgecolor='k', linewidth=0.7)

        # Draw steering angle indicator if not at final position
        if k < len(u_opt):
            phi = u_opt[k, 1]
            # Steering direction from front of car
            front_center = (np.array([p1, p2]) + 
                           (car_length/2) * np.array([np.cos(theta), np.sin(theta)]))
            steer_tip = (front_center + 
                        (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)]))
            ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                    color='crimson', linewidth=1, alpha=alpha+0.1)

    # Mark start and end points
    ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
    ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return fig, ax


# Example usage
if __name__ == "__main__":
    # Parallel parking example
    x_final = (0.5, 0.5, -np.pi/2)
    speed_bounds = (-0.15, 0.6)
    steering_bounds = (-np.pi/8, np.pi/8)
    accel_bound = 0.35
    steering_rate_bound = np.pi/10
    h = 0.1

    print("Car Control Optimization - Parallel Parking")
    print("=" * 50)
    print(f"Target: p1={x_final[0]:.1f}, p2={x_final[1]:.1f}, theta={x_final[2]:.2f} rad")

    x_opt, u_opt = solve_car_control_vectorized(
        x_final,
        speed_bounds=speed_bounds,
        steering_bounds=steering_bounds,
        accel_bound=accel_bound,
        steering_rate_bound=steering_rate_bound
    )

    if x_opt is not None and u_opt is not None:
        print("Optimization successful!")
        print(f"Final: p1={x_opt[-1, 0]:.3f}, p2={x_opt[-1, 1]:.3f}, theta={x_opt[-1, 2]:.3f}")

        # Create combined figure with 3 rows
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

        # Plot 1: Trajectory
        L = 0.1
        car_length = L
        car_width = L * 0.6

        steps_to_show = np.arange(0, len(x_opt), max(1, len(x_opt)//20))
        n_shadows = len(steps_to_show)

        for i, k in enumerate(steps_to_show):
            p1, p2, theta = x_opt[k]
            corners = np.array([
                [ car_length/2,  car_width/2],
                [ car_length/2, -car_width/2],
                [-car_length/2, -car_width/2],
                [-car_length/2,  car_width/2],
                [ car_length/2,  car_width/2],
            ])
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rotated = (R @ corners.T).T + np.array([p1, p2])
            alpha = 0.15 + 0.7 * (i+1)/n_shadows
            ax1.fill(rotated[:,0], rotated[:,1], color='dodgerblue', alpha=alpha, edgecolor='k', linewidth=0.7)

            if k < len(u_opt):
                phi = u_opt[k, 1]
                front_center = (np.array([p1, p2]) +
                               (car_length/2) * np.array([np.cos(theta), np.sin(theta)]))
                steer_tip = (front_center +
                            (car_length/3) * np.array([np.cos(theta + phi), np.sin(theta + phi)]))
                ax1.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                        color='crimson', linewidth=1, alpha=alpha+0.1)

        ax1.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1, 1)
        ax1.set_aspect('equal')

        # Plot 2: First-order controls (Speed and Steering)
        k_steps = np.arange(len(u_opt))
        ax2.plot(k_steps, u_opt[:, 0], 'b-', linewidth=2, label='Speed (m/s)')
        ax2.plot(k_steps, u_opt[:, 1], 'r-', linewidth=2, label='Steering Angle (rad)')
        ax2.set_xlabel('$k$', fontsize=16)
        ax2.set_ylabel('$u_k$', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)

        # Plot 3: Second-order controls (Acceleration and Steering Rate)
        acceleration = np.diff(u_opt[:, 0]) / h
        steering_rate = np.diff(u_opt[:, 1]) / h
        k_steps_diff = np.arange(len(acceleration))
        ax3.plot(k_steps_diff, acceleration, 'g-', linewidth=2, label='Acceleration (m/s²)')
        ax3.plot(k_steps_diff, steering_rate, 'm-', linewidth=2, label='Steering Rate (rad/s)')
        ax3.set_xlabel('$k$', fontsize=16)
        ax3.set_ylabel(r'$\frac{u_k - u_{k-1}}{h}$', fontsize=18)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=12)

        plt.tight_layout()
        fig.savefig("parallel_parking.pdf", bbox_inches="tight", dpi=300)
        print("Saved parallel_parking.pdf")
        plt.show()
    else:
        print("Optimization failed!")

    print("\nDone!")
