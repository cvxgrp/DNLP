


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

import cvxpy as cp

rng = np.random.default_rng(3)

# number of circles
n = 3
radius = rng.uniform(1.0, 3.0, n)

# build problem
centers = cp.Variable((2, n), name='c')
constraints = []
for i in range(n - 1):
    for j in range(i + 1, n):
        constraints += [cp.sum(cp.square(centers[:, i] - centers[:, j])) >=
                         (radius[i] + radius[j]) ** 2]

# initialize centers to random locations
centers.value = rng.uniform(-5.0, 5.0, (2, n))

#t = cp.Variable()
#obj = cp.Minimize(t)
#constraints += [cp.max(cp.norm_inf(centers, axis=0) + radius) <= t]

obj = cp.Minimize(cp.max(cp.max(cp.abs(centers), axis=0) + radius))
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none',
              least_square_init_duals='no')


# the original 'c' variable
circle_centers = centers.value

max_expr = cp.max(cp.max(cp.abs(circle_centers), axis=0) + radius)

assert max_expr.value is not None, (
    "Optimization did not assign a value to the variable."
)

square_size = float(max_expr.value) * 2
pi = np.pi
ratio = pi * np.sum(np.square(radius)) / (square_size**2)

# create plot to visualize the packing
fig, ax = plt.subplots(figsize=(4, 4))
ax.set_aspect("equal", adjustable="box")
fig.set_dpi(150)

# draw circles
for i in range(n):
    x_val = circle_centers[0, i]
    y_val = circle_centers[1, i]
    if x_val is None or y_val is None:
        msg = f"Circle center value not assigned for index {i}."
        raise ValueError(msg)
    circle = Circle(
        (float(x_val), float(y_val)),  # (x, y) center
        radius[i],  # radius
        fill=False,  # outline only
        ec="b",
        lw=1.2,  # edge color/width
    )
    ax.add_patch(circle)

# draw square border
border = Rectangle(
    (-square_size / 2, -square_size / 2),  # bottom-left
    square_size,
    square_size,  # width, height
    fill=False,
    ec="g",
    lw=1.5,
)
ax.add_patch(border)

# limits and cosmetics
ax.set_xlim(float(-square_size / 2), float(square_size / 2))
ax.set_ylim(float(-square_size / 2), float(square_size / 2))
ax.set_xlabel("x")
ax.set_ylabel("y")
#ax.set_title(f"Circle packing (ratio={ratio:.3f})")
print(f"Circle packing (ratio={ratio:.3f})")
plt.savefig("circle_packing_working_formulation.pdf")

