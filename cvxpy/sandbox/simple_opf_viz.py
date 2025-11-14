import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags

import cvxpy as cp


def solve_and_visualize_opf():
    """Solve OPF and create beautiful network visualizations using matplotlib only."""
    
    # Problem setup (same as original)
    N = 9
    
    # Real generation bounds
    P_Gen_lb = np.zeros(N)
    P_Gen_lb[[0, 1, 2]] = [10, 10, 10]
    P_Gen_ub = np.zeros(N)
    P_Gen_ub[[0, 1, 2]] = [250, 300, 270]
    
    # Reactive generation bounds
    Q_Gen_lb = np.zeros(N)
    Q_Gen_lb[[0, 1, 2]] = [-5, -5, -5]
    Q_Gen_ub = np.zeros(N)
    Q_Gen_ub[[0, 1, 2]] = [300, 300, 300]
    
    # Power demand
    P_Demand = np.zeros(N)
    P_Demand[[4, 6, 8]] = [54, 60, 75]
    Q_Demand = np.zeros(N)
    Q_Demand[[4, 6, 8]] = [18, 21, 30]
    
    # Branch data
    branch_data = np.array([
        [0, 3, 0.0, 0.0576, 0.0],
        [3, 4, 0.017, 0.092, 0.158],
        [5, 4, 0.039, 0.17, 0.358],
        [2, 5, 0.0, 0.0586, 0.0],
        [5, 6, 0.0119, 0.1008, 0.209],
        [7, 6, 0.0085, 0.072, 0.149],
        [1, 7, 0.0, 0.0625, 0.0],
        [7, 8, 0.032, 0.161, 0.306],
        [3, 8, 0.01, 0.085, 0.176],
    ])
    
    M = branch_data.shape[0]
    base_MVA = 100
    
    # Build incidence matrix A
    from_bus = branch_data[:, 0].astype(int)
    to_bus = branch_data[:, 1].astype(int)
    A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
        csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))
    
    # Network impedance
    z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA
    
    # Bus admittance matrix
    Y_0 = A @ diags(1.0 / z) @ A.T
    y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
    Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
    Y_sh = diags(Y_sh_diag)
    Y = Y_0 + Y_sh
    Y_dense = Y.toarray()
    
    G = np.real(Y_dense)
    B = np.imag(Y_dense)
    
    # Variables
    V_mag = cp.Variable(N, bounds=[0.9, 1.1])
    V_ang = cp.Variable(N)
    P_G = cp.Variable(N, bounds=[P_Gen_lb, P_Gen_ub])
    Q_G = cp.Variable(N, bounds=[Q_Gen_lb, Q_Gen_ub])
    
    # Initialize
    V_mag.value = np.ones(N)
    V_ang.value = np.zeros(N)
    P_G.value = (P_Gen_lb + P_Gen_ub) / 2
    Q_G.value = (Q_Gen_lb + Q_Gen_ub) / 2
    
    # Constraints
    constraints = []
    constraints.append(V_ang[0] == 0)
    
    # Power flow equations
    V_ang_col = cp.reshape(V_ang, (N, 1), order='F')
    V_ang_row = cp.reshape(V_ang, (1, N), order='F')
    theta_diff = V_ang_col - V_ang_row
    
    cos_theta = cp.cos(theta_diff)
    sin_theta = cp.sin(theta_diff)
    
    real_coeffs = cp.multiply(G, cos_theta) + cp.multiply(B, sin_theta)
    reactive_coeffs = cp.multiply(G, sin_theta) - cp.multiply(B, cos_theta)
    
    P_injection = cp.multiply(V_mag, real_coeffs @ V_mag)
    Q_injection = cp.multiply(V_mag, reactive_coeffs @ V_mag)
    
    constraints.append(P_G - P_Demand == P_injection)
    constraints.append(Q_G - Q_Demand == Q_injection)
    
    # Objective
    objective = cp.Minimize(
        0.11 * P_G[0]**2 + 5 * P_G[0] + 150 +
        0.085 * P_G[1]**2 + 1.2 * P_G[1] + 600 +
        0.1225 * P_G[2]**2 + P_G[2] + 335
    )
    
    # Solve with IPOPT (required for nonlinear AC OPF)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.IPOPT, verbose=True, nlp=True)
    
    print(f"Solver status: {problem.status}")
    print(f"Optimal objective value: {problem.value:.2f}")
    
    # Compute line flows
    def compute_line_flows(V_mag_val, V_ang_val):
        flows = {}
        for i, (fb, tb, r, x, b) in enumerate(branch_data):
            fb, tb = int(fb), int(tb)
            z_line = complex(r, x) / base_MVA
            y_line = 1 / z_line if z_line != 0 else 1e6
            y_shunt = complex(0, b * base_MVA / 2)
            
            V_from = V_mag_val[fb] * np.exp(1j * V_ang_val[fb])
            V_to = V_mag_val[tb] * np.exp(1j * V_ang_val[tb])
            
            I_from_to = y_line * (V_from - V_to) + y_shunt * V_from
            S_from_to = V_from * np.conj(I_from_to)
            
            flows[i] = {
                'from_bus': fb, 'to_bus': tb,
                'P_from_to': np.real(S_from_to),
                'Q_from_to': np.imag(S_from_to),
                'flow_magnitude': abs(S_from_to)
            }
        return flows
    
    flows = compute_line_flows(V_mag.value, V_ang.value)
    
    # Create visualizations
    create_power_flow_plots(V_mag.value, V_ang.value, P_G.value, Q_G.value, 
                           P_Demand, Q_Demand, flows, branch_data, N)
    
    return problem, flows

def create_power_flow_plots(V_mag, V_ang, P_G, Q_G, P_Demand, Q_Demand, flows, branch_data, N):
    """Create real power flow visualization only."""
    
    # Create NetworkX graph
    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    
    for i, (from_bus, to_bus, r, x, b) in enumerate(branch_data):
        G.add_edge(int(from_bus), int(to_bus), branch_id=i, resistance=r, reactance=x)
    
    # Try different layout for better label spacing
    # First try circular layout as base
    pos_circular = nx.circular_layout(G)
    
    # Then apply spring layout with the circular positions as starting point
    pos = nx.spring_layout(G, pos=pos_circular, k=2.0, iterations=100, seed=42)
    
    # Alternative: try kamada_kawai layout which often produces good spacing
    # pos = nx.kamada_kawai_layout(G)
    
    # Scale up the layout to reduce label overlap
    scale_factor = 1.3
    pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in pos.items()}
    
    # Manual adjustment for node 9 (index 8) to avoid overlap - move it to the right
    if 8 in pos:  # Node 9 has index 8
        current_x, current_y = pos[8]
        pos[8] = (current_x + 0.15, current_y)  # Move 0.4 units to the right
    
    # Create figure with single plot for real power flows - optimized for PDF
    # Use larger figure size for full page PDF (11x8.5 inches, landscape orientation)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=300)  # High DPI for quality
    fig.suptitle('Real Power Flows (MW)', fontsize=18, fontweight='bold', y=0.95)
    
    # Real power flows only
    plot_power_flows(ax, G, pos, flows, 'P', N, P_G, P_Demand)
    
    # Adjust layout for PDF - maximize plot area
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
    
    # Save as high-quality PDF
    pdf_filename = 'power_flow_visualization.pdf'
    plt.savefig(pdf_filename, 
                format='pdf',
                dpi=300,  # High resolution
                bbox_inches='tight',  # Remove extra whitespace
                facecolor='white',
                edgecolor='none',
                orientation='landscape')  # Landscape for better network layout
    
    print(f"High-quality PDF saved as: {pdf_filename}")
    
    # Also display the plot
    plt.show()

def plot_power_flows(ax, G, pos, flows, power_type, N, P_G=None, P_Demand=None):
    """Plot power flows with colored and sized edges."""
    # Get flow values
    if power_type == 'P':
        flow_values = [flows[i]['P_from_to'] for i in flows.keys()]
        cmap = 'Blues'  # Light blue to dark blue gradient
    else:
        flow_values = [flows[i]['Q_from_to'] for i in flows.keys()]
        cmap = 'Blues'  # Light blue to dark blue gradient
    
    flow_values = np.array(flow_values)
    
    # Identify generator and load buses
    gen_buses = []
    load_buses = []
    regular_buses = []
    
    # Check each bus to categorize it (only if P_G and P_Demand are provided)
    for i in range(N):
        is_gen = False
        is_load = False
        
        # Check if it's a generator bus (has generation > 1 MW)
        if P_G is not None and i < len(P_G) and P_G[i] > 1:
            gen_buses.append(i)
            is_gen = True
        
        # Check if it's a load bus (has demand > 1 MW)  
        if P_Demand is not None and i < len(P_Demand) and P_Demand[i] > 1:
            load_buses.append(i)
            is_load = True
            
        # If neither generator nor load, it's a regular bus
        if not is_gen and not is_load:
            regular_buses.append(i)
    
    # Draw nodes with different shapes for generators, loads, and regular buses
    # Regular buses (circles)
    if regular_buses:
        nx.draw_networkx_nodes(G, pos, nodelist=regular_buses, ax=ax, 
                              node_color='lightgray', node_size=1000, 
                              node_shape='o', alpha=0.8, edgecolors='black', linewidths=2)
    
    # Generator buses (squares)
    if gen_buses:
        nx.draw_networkx_nodes(G, pos, nodelist=gen_buses, ax=ax, 
                              node_color='lightgreen', node_size=1000, 
                              node_shape='s', alpha=0.8, edgecolors='darkgreen', linewidths=3)
    
    # Load buses (diamonds)
    if load_buses:
        nx.draw_networkx_nodes(G, pos, nodelist=load_buses, ax=ax, 
                              node_color='lightcoral', node_size=1000, 
                              node_shape='D', alpha=0.8, edgecolors='darkred', linewidths=3)
    
    # Add bus labels
    nx.draw_networkx_labels(G, pos, {i: str(i+1) for i in range(N)}, ax=ax, 
                           font_size=16, font_weight='bold')
    
    # Add demand labels under load buses
    if P_Demand is not None:
        for i in load_buses:
            if i < len(P_Demand) and P_Demand[i] > 1:
                x, y = pos[i]
                # Position demand label below the node
                ax.text(x, y - 0.12, f'{P_Demand[i]:.0f} MW', 
                       ha='center', va='top', fontsize=11, 
                       fontweight='bold', color='darkred',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                alpha=0.8, edgecolor='darkred'))
    
    # Draw edges with flow-based styling
    for i, flow_data in flows.items():
        from_bus, to_bus = flow_data['from_bus'], flow_data['to_bus']
        flow_val = flow_values[i]
        abs_flow_val = abs(flow_val)  # Use absolute value for positive flows only
        
        # Edge width and color based on absolute flow (using 0 to 150 scale)
        # Increased width variation: min=5, max=10 (instead of 3 to 6)
        width = 5 + 5 * abs_flow_val / 150.0
        # Map absolute flow values to colormap: start at 0.5 (medium blue) to 0.9 (darker blue)
        color_intensity = 0.5 + 0.4 * (abs_flow_val / 150.0)  # Scale from 0.5 to 0.9
        # Clamp color intensity to [0.5, 0.9] range for smaller color variation
        color_intensity = max(0.5, min(0.9, color_intensity))
        color = plt.cm.get_cmap(cmap)(color_intensity)
        
        nx.draw_networkx_edges(G, pos, edgelist=[(from_bus, to_bus)],
                              width=width, edge_color=[color], ax=ax, alpha=0.8)
        
        # Add small directional markers (triangles) close to destination node
        x1, y1 = pos[from_bus]
        x2, y2 = pos[to_bus]
        
        # Determine flow direction and marker position
        if flow_val >= 0:  # Positive flow: from_bus → to_bus
            marker_pos = 0.85  # 85% along the edge
            marker_x = x1 + (x2 - x1) * marker_pos
            marker_y = y1 + (y2 - y1) * marker_pos
            # Calculate angle for triangle orientation
            angle = np.arctan2(y2 - y1, x2 - x1)
        else:  # Negative flow: to_bus → from_bus
            marker_pos = 0.15  # 15% along the edge
            marker_x = x1 + (x2 - x1) * marker_pos
            marker_y = y1 + (y2 - y1) * marker_pos
            # Calculate angle for triangle orientation (reversed)
            angle = np.arctan2(y1 - y2, x1 - x2)
        
        # Draw triangle marker (original size)
        triangle_size = 0.04
        triangle_x = [
            marker_x + triangle_size * np.cos(angle),
            marker_x + triangle_size * np.cos(angle + 2.6),
            marker_x + triangle_size * np.cos(angle - 2.6)
        ]
        triangle_y = [
            marker_y + triangle_size * np.sin(angle),
            marker_y + triangle_size * np.sin(angle + 2.6),
            marker_y + triangle_size * np.sin(angle - 2.6)
        ]
        
        ax.fill(triangle_x, triangle_y, color='black', alpha=0.9, zorder=3)
        
        # Add flow labels (showing absolute values)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{abs_flow_val:.1f}', ha='center', va='center', 
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    ax.axis('off')
    
    # Create legend for node types
    from matplotlib.patches import Patch
    
    legend_elements = []
    if gen_buses:
        legend_elements.append(Patch(facecolor='lightgreen', edgecolor='darkgreen', 
                                   label='Generator Buses'))
    if load_buses:
        legend_elements.append(Patch(facecolor='lightcoral', edgecolor='darkred', 
                                   label='Load Buses'))
    if regular_buses:
        legend_elements.append(Patch(facecolor='lightgray', edgecolor='black', 
                                   label='Transit Buses'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

    # Colorbar with fixed 0 to 130 range
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 130))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Real Power (MW)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

if __name__ == "__main__":
    print("=" * 60)
    print("SOLVING OPTIMAL POWER FLOW WITH NETWORK VISUALIZATION")
    print("=" * 60)
    
    problem, flows = solve_and_visualize_opf()
    
    print("\n" + "=" * 60)
    print("FLOW SUMMARY")
    print("=" * 60)
    print(f"{'Line':<6} {'From':<6} {'To':<6} {'P (MW)':<10} {'Q (MVAr)':<12} {'|S| (MVA)':<12}")
    print("-" * 60)
    
    for line_id, flow in flows.items():
        print(f"{line_id+1:<6} {flow['from_bus']+1:<6} {flow['to_bus']+1:<6} "
              f"{flow['P_from_to']:<10.2f} {flow['Q_from_to']:<12.2f} "
              f"{flow['flow_magnitude']:<12.2f}")
    
    print("\nVisualization Features:")
    print("- Node size and color indicate voltage magnitude")
    print("- Edge thickness represents power flow magnitude") 
    print("- Edge color indicates flow direction (blue→red = negative→positive)")
    print("- Generation shown as green arrows, demand as red arrows")
    print("- All values labeled for detailed analysis")
