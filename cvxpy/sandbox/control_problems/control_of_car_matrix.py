import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def solve_car_control_vectorized(x_final, L=0.1, N=50, h=0.1, gamma=10):
    """
    Solve the nonlinear optimal control problem for car trajectory planning.
    
    Parameters:
    - x_final: tuple (p1, p2, theta) for final position and orientation
    - L: wheelbase length
    - N: number of time steps
    - h: time step size
    - gamma: weight for control smoothness term
    
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

    constraints.append(x_next[:, 0] == x_curr[:, 0] + h * cp.multiply(v, cp.cos(theta)))
    constraints.append(x_next[:, 1] == x_curr[:, 1] + h * cp.multiply(v, cp.sin(theta)))
    constraints.append(x_next[:, 2] == x_curr[:, 2] + h * cp.multiply(v / L, cp.tan(delta)))

    # speed limit bounds
    constraints += [u[:, 0] >= -1.0, u[:, 0] <= 0.25]
    # steering angle bounds
    constraints += [u[:, 1] >= -np.pi / 4, u[:, 1] <= np.pi / 4]
    # acceleration bounds
    #constraints += [cp.abs(u[1:, 0] - u[:-1, 0]) <= 0.5 * h]
    # steering angle rate bounds
    #constraints += [cp.abs(u[1:, 1] - u[:-1, 1]) <= (np.pi / 4) * h]
    # Create and solve the problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
    return x.value, u.value

def plot_trajectory(x_opt, u_opt, L, h, title="Car Trajectory"):
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
            front_center = np.array([p1, p2]) + (car_length/2) * np.array([np.cos(theta), np.sin(theta)])
            steer_tip = front_center + (car_length/2) * np.array([np.cos(theta + phi), np.sin(theta + phi)])
            ax.plot([front_center[0], steer_tip[0]], [front_center[1], steer_tip[1]],
                    color='crimson', linewidth=2, alpha=alpha+0.1)

    # Mark start and end points
    ax.plot(x_opt[0, 0], x_opt[0, 1], 'go', markersize=10, label='Start')
    ax.plot(x_opt[-1, 0], x_opt[-1, 1], 'ro', markersize=10, label='Goal')

    ax.set_xlabel('p1')
    ax.set_ylabel('p2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    return fig, ax


def plot_acceleration_and_metrics(x_opt, u_opt, h, title="Vehicle Dynamics Analysis"):
    """
    Plot acceleration, orientation, and other important metrics throughout the trajectory.
    
    Parameters:
    - x_opt: optimal states (N+1 x 3) containing [p1, p2, theta]
    - u_opt: optimal controls (N x 2) containing [speed, steering_angle]
    - h: time step size
    - title: plot title
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time vector
    time_steps = np.arange(len(u_opt)) * h
    
    # Extract speed and steering angle from controls
    speed = u_opt[:, 0]
    steering_angle = u_opt[:, 1]
    
    # Calculate acceleration (derivative of speed)
    acceleration = np.zeros_like(speed)
    if len(speed) > 1:
        acceleration[1:] = np.diff(speed) / h
        acceleration[0] = acceleration[1]  # Use second value for first point
    
    # Extract orientation from states
    orientation = x_opt[:-1, 2]  # theta values (excluding last one to match control length)
    
    # Calculate angular velocity (derivative of orientation)
    angular_velocity = np.zeros_like(orientation)
    if len(orientation) > 1:
        angular_velocity[1:] = np.diff(orientation) / h
        angular_velocity[0] = angular_velocity[1]  # Use second value for first point
    
    # Plot 1: Acceleration
    ax1.plot(time_steps, acceleration, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Longitudinal Acceleration')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Orientation (angle)
    ax2.plot(time_steps, np.degrees(orientation), 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Orientation (degrees)')
    ax2.set_title('Vehicle Heading')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed profile
    ax3.plot(time_steps, speed, 'g-', linewidth=2, marker='^', markersize=4)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Speed Profile')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 4: Steering angle
    ax4.plot(time_steps, np.degrees(steering_angle), 'purple', 
             linewidth=2, marker='d', markersize=4)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Steering Angle (degrees)')
    ax4.set_title('Steering Input')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


# Example usage
if __name__ == "__main__":
    # Test cases from the figure
    test_cases = [
        #((0, 1, 0), "Move forward to (0, 1)"),
        #((0, 1, np.pi/2), "Move to (0, 1) and turn 90°"),
        ((0, 0.5, 0), "Move forward to (0, 0.5)"),
        #((0.5, 0.5, -np.pi/2), "Move to (0.5, 0.5) and turn -90°")
    ]
    
    # Solve for each test case
    for x_final, description in test_cases:
        print(f"\nSolving for: {description}")
        print(f"Target state: p1={x_final[0]}, p2={x_final[1]}, theta={x_final[2]:.2f}")
        
        try:
            x_opt, u_opt = solve_car_control_vectorized(x_final)
            
            if x_opt is not None and u_opt is not None:
                print("Optimization successful!")
                print(
                    f"Final position: p1={x_opt[-1, 0]:.3f}, "
                    f"p2={x_opt[-1, 1]:.3f}, "
                    f"theta={x_opt[-1, 2]:.3f}"
                )
                
                # Plot the trajectory
                fig, ax = plot_trajectory(x_opt, u_opt, L=0.1, h=0.1, title=description)
                plt.show()
                
                # Plot control inputs only
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                time_steps = np.arange(len(u_opt)) * 0.1  # h = 0.1
                
                # Speed plot
                ax1.plot(time_steps, u_opt[:, 0], 'b-', linewidth=2, marker='o', markersize=3)
                ax1.set_ylabel('Speed (m/s)')
                ax1.set_title('Control Input: Speed')
                ax1.grid(True, alpha=0.3)
                ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                # Steering angle plot
                ax2.plot(time_steps, np.degrees(u_opt[:, 1]), 'r-', 
                         linewidth=2, marker='s', markersize=3)
                ax2.set_ylabel('Steering Angle (degrees)')
                ax2.set_xlabel('Time (s)')
                ax2.set_title('Control Input: Steering Angle')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                plt.show()
            else:
                print("Optimization failed!")
                
        except Exception as e:
            print(f"Error: {e}")
"""
    # Additional analysis: plot control inputs
    if x_opt is not None and u_opt is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        time_steps = np.arange(len(u_opt)) * 0.1  # h = 0.1
        
        ax1.plot(time_steps, u_opt[:, 0], 'b-', linewidth=2)
        ax1.set_ylabel('Speed s(t)')
        ax1.set_xlabel('Time')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time_steps, u_opt[:, 1], 'r-', linewidth=2)
        ax2.set_ylabel('Steering angle φ(t)')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
"""
