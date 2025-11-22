import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

G = nx.DiGraph()

edges = [
    (0, 3),
    (3, 4),
    (3, 8),
    (8, 7),
    (1, 7),
    (5, 4),
    (5, 6),
    (2, 5),
]

G.add_edges_from(edges)

flows = {
    (0,3): 10,
    (3,4): 7,
    (3,8): 4,
    (8,7): 3,
    (1,7): 1,
    (5,4): 6,
    (5,6): 2,
    (2,5): 5,
}

pos = {
    0: (0, 0),
    3: (2, 0),
    4: (4, 0),

    8: (2, -1.5),
    7: (1, -3),
    1: (0, -3),

    5: (4, -1.5),
    6: (4, -3),
    2: (5, -1.5),
}

flow_values = np.array([flows[e] for e in edges], dtype=float)
max_flow = flow_values.max()
line_widths = 2 + 4 * (flow_values / max_flow)

# --- Create figure/axis ---
fig, ax = plt.subplots(figsize=(7, 6))

# --- Draw nodes and labels ---
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900)
nx.draw_networkx_labels(G, pos, ax=ax)

# --- Build a LineCollection for edges (color + width) ---
segments = []
for (u, v) in edges:
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    segments.append([(x1, y1), (x2, y2)])

lc = LineCollection(
    segments,
    array=flow_values,
    cmap="plasma",
    linewidths=line_widths,
)

ax.add_collection(lc)

# --- Draw flow labels next to edges ---
flow_labels = {e: str(flows[e]) for e in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=flow_labels, ax=ax)

# --- Add colorbar (NOW works) ---
fig.colorbar(lc, ax=ax, label="Flow Amount")

ax.set_axis_off()
plt.show()
