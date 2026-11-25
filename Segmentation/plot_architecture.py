import matplotlib.pyplot as plt
import numpy as np
import os
try:
    import networkx as nx
except ImportError:
    os.system("pip install networkx")
    import networkx as nx

def plot_network_path(path, title, filename):
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Levels: 0=4, 1=8, 2=16, 3=32
    layers = np.arange(1, len(path) + 1)
    
    # Plot grid
    for y in [0, 1, 2, 3]:
        ax.plot(layers, [y]*len(layers), 'o', color='lightgray', markersize=10)
        
    # Plot path
    ax.plot(layers, path, '-o', color='blue', markersize=12, linewidth=2, markerfacecolor='royalblue')
    
    # Formatting
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['4', '8', '16', '32'])
    ax.set_xticks(layers)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Downsample Factor')
    ax.invert_yaxis()
    #ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def parse_cell(cell_matrix):
    PRIMITIVES = [
        'none', 'max_pool_3x3', 'avg_pool_3x3', 'skip_connect',
        'sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5'
    ]
    edges = []
    # decode edge indices to (source, target)
    for j, op_idx in cell_matrix:
        # Find which step 'j' belongs to
        start = 0
        n = 2
        step = 0
        while True:
            end = start + n
            if start <= j < end:
                source = j - start
                target = step + 2
                break
            start = end
            n += 1
            step += 1
        
        op_name = PRIMITIVES[op_idx]
        edges.append((source, target, op_name))
    return edges

def plot_cell_structure(cell_matrix, title, filename):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
    import numpy as np
    
    edges = parse_cell(cell_matrix)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')
    
    # Coordinates mapping
    node_coords = {}
    node_coords[0] = (0, 0)
    node_coords[1] = (2.5, 0)
    
    # Spread intermediate nodes widely in X and stagger in Y
    y_positions = [3.5, -3.5, 1.5, -1.5, 0]
    for i in range(2, 7):
        node_coords[i] = (6 + (i-2)*2.5, y_positions[i-2])
        
    concat_x = 18
    out_x = 21
    
    def draw_box(ax, center, text, width=1.5, height=0.8, facecolor='#e0e0e0'):
        x, y = center
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                         facecolor=facecolor, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=16, zorder=4)
        
    draw_box(ax, node_coords[0], "$H^{i-2}$")
    draw_box(ax, node_coords[1], "$H^{i-1}$")
    
    # Cloud between inputs
    ax.text(1.25, 0, "...", ha='center', va='center', fontsize=24, zorder=4)
    
    # Draw intermediate nodes as + circles
    for i in range(2, 7):
        x, y = node_coords[i]
        circle = Circle((x, y), radius=0.3, facecolor='white', edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, "+", ha='center', va='center', fontsize=18, zorder=4)
        
    # Draw concat block
    rect_concat = Rectangle((concat_x - 0.4, -4.5), 0.8, 9, facecolor='white', edgecolor='black', linewidth=1.5, zorder=3)
    ax.add_patch(rect_concat)
    ax.text(concat_x, 0, "concat", ha='center', va='center', rotation=90, fontsize=16, zorder=4)
    
    # Output node
    draw_box(ax, (out_x, 0), "$H^i$")
    
    # Connect intermediate nodes to concat block
    for i in range(2, 7):
        x, y = node_coords[i]
        arrow = FancyArrowPatch((x+0.3, y), (concat_x-0.4, y),
                                connectionstyle="arc3,rad=0",
                                arrowstyle="-|>", mutation_scale=20, color='black', linewidth=1.5, zorder=1)
        ax.add_patch(arrow)
        
    # Connect concat to output
    arrow = FancyArrowPatch((concat_x+0.4, 0), (out_x-0.75, 0),
                            connectionstyle="arc3,rad=0",
                            arrowstyle="-|>", mutation_scale=20, color='black', linewidth=1.5, zorder=1)
    ax.add_patch(arrow)
    
    def get_bezier_point(x1, y1, x2, y2, rad, t=0.5):
        # matplotlib arc3 places control point at distance rad * length from chord midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        cx = mid_x - dy * rad
        cy = mid_y + dx * rad
        bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
        by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
        return bx, by

    # Draw operations (edges)
    for idx, (src, tgt, op) in enumerate(edges):
        if op == 'none': continue
        
        x1, y1 = node_coords[src]
        x2, y2 = node_coords[tgt]
        
        r1 = 0.75 if src < 2 else 0.3
        r2 = 0.3
        
        # Determine curvature based on src/tgt distance to separate overlapping lines
        rad = 0.15 + (tgt - src) * 0.05
        if src == 0: rad = 0.25 + (tgt - 2) * 0.05
        if y2 < y1: rad = -rad
        
        # Add slight variation for edges sharing similar paths
        rad += (idx % 3) * 0.05

        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="-|>", mutation_scale=20, color='black', linewidth=1.5, 
                                shrinkA=r1*30, shrinkB=r2*30, zorder=1)
        ax.add_patch(arrow)
        
        # Jitter the t value to spread text boxes along the curve
        # t = 0.5 is middle. Spread between 0.35 and 0.65
        t = 0.5 + ((src + tgt + idx) % 3 - 1) * 0.15
        box_x, box_y = get_bezier_point(x1, y1, x2, y2, rad, t)
        
        # Colors to match paper
        if 'sep' in op: color = '#fff2cc' # light yellow
        elif 'dil' in op or 'atr' in op: 
            color = '#f8cecc' # light pink
            op = op.replace('dil', 'atr')
        elif 'pool' in op: color = '#dae8fc' # light blue
        else: color = 'white'
            
        op_text = op.replace('_conv_', '\n').replace('_pool_', '\n').replace('skip_connect', 'skip\nconnect')
            
        ax.text(box_x, box_y, op_text, ha='center', va='center', fontsize=12,
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='square,pad=0.4'),
                zorder=5)
                
    ax.set_xlim(-1, 23)
    ax.set_ylim(-5, 5)
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Standard Results
    path_std = [1, 1, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3]
    cell_std = [[1, 3], [0, 0], [3, 5], [4, 3], [5, 4], [6, 1], [12, 3], [13, 5], [14, 4], [17, 5]]
    
    # Modified (CVAR) Results
    path_mod = [0, 1, 2, 2, 3, 3, 3, 2, 3, 2, 1, 1]
    cell_mod = [[1, 4], [0, 0], [3, 4], [2, 4], [6, 7], [5, 7], [9, 6], [11, 4], [19, 5], [14, 3]]
    
    # Plotting
    plot_network_path(path_std, "Standard AutoDeepLab - Network Path", "network_path_standard.png")
    plot_network_path(path_mod, "Modified CVAR - Network Path", "network_path_modified.png")
    
    # plot_cell_structure(cell_std, "Standard AutoDeepLab - Cell Structure", "cell_structure_standard.png")
    # plot_cell_structure(cell_mod, "Modified CVAR - Cell Structure", "cell_structure_modified.png")
    print("Saved all 4 images successfully in the current directory.")
