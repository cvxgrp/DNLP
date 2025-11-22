import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# --- Flows: assume flows are provided in this dictionary (replace with your dict) ---
flows = {
    (0, 3): 10.0,
    (3, 4): 4.534,
    (3, 8): 5.4347,
    (7, 8): 69.57,
    (1, 7): 125.368,
    (5, 4): 49.466,
    (5, 6): 6.398,
    (2, 5): 57.03,
    (7, 6): 53.6,
}


losses = {
    (0, 3): 0.00,
    (3, 4): 0.01,
    (3, 8): 0.02,
    (7, 8): 1.91,
    (1, 7): 0.00,
    (5, 4): 1.14,
    (5, 6): 0.03,
    (2, 5): 0.00,
    (7, 6): 0.29,
}

# Demands at sink nodes (units matching your flows). Edit these values as needed.
demands = {
    4: 54.0,
    6: 60.0,
    8: 75.0,
}

G = nx.DiGraph()

# node drawing size (points^2 as used by scatter)
node_size = 900

# Add edges
edges = [
    (0, 3),
    (3, 4),

    (3, 8),
    (7, 8),
    (1, 7),

    (5, 4),
    (5, 6),
    (2, 5),
    (7, 6),
]

G.add_edges_from(edges)

# Manual layout (x, y)
pos = {
    0: (0, 0),
    3: (2, 0),
    4: (4, 0),

    8: (2, -1.5),
    7: (2, -3),
    1: (0, -3),

    5: (4, -1.5),
    6: (4, -3),
    2: (5, -1.5),
}

# Build a list of flow values in the same order as `edges`
flow_values = np.array([flows.get(e, 0.0) for e in edges], dtype=float)
max_flow = flow_values.max() if flow_values.size > 0 else 1.0

# Make all edges the same thickness and color
edge_width = 3

# --- Create figure/axis ---
fig, ax = plt.subplots(figsize=(8, 6))

# Draw nodes with shapes for generator/sink/other
# Generators: nodes 0,1,2 -> squares
gen_nodes = [0, 1, 2]
# Sinks: nodes 8,4,6 -> diamonds
sink_nodes = [8, 4, 6]
demands = {4: 54, 6: 60, 8: 75}
# Other nodes
other_nodes = [n for n in G.nodes() if n not in gen_nodes + sink_nodes]

# Draw other nodes (circles)
# Draw other nodes (circles)
nx.draw_networkx_nodes(
    G, pos, nodelist=other_nodes, node_shape="o", node_color="#9ecae1", node_size=node_size, ax=ax
)
# Draw generator nodes (squares)
# Draw generator nodes (squares)
nx.draw_networkx_nodes(
    G, pos, nodelist=gen_nodes, node_shape="s", node_color="#31a354", node_size=node_size, ax=ax
)
# Draw sink nodes (diamonds)
# Draw sink nodes (diamonds)
nx.draw_networkx_nodes(
    G, pos, nodelist=sink_nodes, node_shape="D", node_color="#fb6a4a", node_size=node_size, ax=ax
)

# Draw labels on top of nodes
nx.draw_networkx_labels(G, pos, ax=ax)

# Annotate sink nodes with demand values (placed to the right of the marker)
for n in sink_nodes:
    if n in demands:
        x, y = pos[n]
        # offset in points: move right and slightly up so it doesn't overlap the node
        ax.annotate(
            f"demand = {demands[n]:.1f}",
            xy=(x, y),
            xytext=(25, 6),
            textcoords="offset points",
            fontsize=11,
            ha="left",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2),
        )

# Draw edges as shortened arrows so they don't underlap node markers.
# We'll shorten each edge by the node marker radius (in display points).
def shorten_point(a, b, offset_pixels):
    """Return a point moved from a towards b by offset_pixels (in display px)."""
    # transform to display (pixel) coords
    a_disp = ax.transData.transform(a)
    b_disp = ax.transData.transform(b)
    vec = b_disp - a_disp
    dist = np.hypot(vec[0], vec[1])
    if dist == 0:
        return a
    # unit vector
    u = vec / dist
    new_a_disp = a_disp + u * offset_pixels
    # back to data coords
    return ax.transData.inverted().transform(new_a_disp)

# compute node radius in display pixels from node_size (points^2)
# approximate marker radius in points = sqrt(s) / 2
radius_pts = np.sqrt(node_size) / 2.0
radius_pixels = radius_pts * (fig.dpi / 72.0)

for (u, v) in edges:
    start = np.array(pos[u])
    end = np.array(pos[v])
    # shorten start and end by radius_pixels so arrows stop at node edge
    new_start = shorten_point(start, end, radius_pixels)
    new_end = shorten_point(end, start, radius_pixels)
    arrow = FancyArrowPatch(
        posA=new_start,
        posB=new_end,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=edge_width,
        color="#444444",
        zorder=1,
    )
    ax.add_patch(arrow)

# Add numeric flow labels at the midpoint of each edge with a small perpendicular offset
for (u, v), f in zip(edges, flow_values):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        offx, offy = 0.0, 0.0
    else:
        # perpendicular unit vector
        ux, uy = -dy / length, dx / length
        # offset magnitude (tune as needed)
        offset = 0.12
        offx, offy = ux * offset, uy * offset

    # loss: either provided in `losses` dict or compute default I^2R-style loss
    if (u, v) in losses:
        loss = losses[(u, v)]
    else:
        # default: small coefficient * flow^2 (tweak coefficient to match units)
        loss = 0.005 * (f ** 2)

    label = f"{f:.2f}\n({loss:.2f})"

    # For certain vertical branches, move the text 0.5 cm to the right.
    # 0.5 cm ≈ 14 points (1 point = 1/72 inch, 1 cm = 28.3464567 points)
    vertical_branches = {(3, 8), (5, 4), (5, 6), (7, 8)}
    if (u, v) in vertical_branches:
        x_offset_pts = 14  # shift right by ~0.5 cm
        if (u, v) == (7, 8) or (u, v) == (5, 4): 
            x_offset_pts *= 2 
    else:
        x_offset_pts = 0

    # Use annotate: place text at data coords (mx+offx, my+offy) and apply an
    # additional offset in points to the right for the specified branches.
    ax.annotate(
        label,
        xy=(mx + offx, my + offy),
        xytext=(x_offset_pts, 0),
        xycoords="data",
        textcoords="offset points",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.3),
    )

# No colorbar: edges are drawn with a single color now

ax.set_axis_off()

# Save output image so running non-interactively still produces a viewable file
out_file = "power_flow.pdf"
fig.tight_layout()
fig.savefig(out_file, dpi=150)
print(f"Saved plot to {out_file}")