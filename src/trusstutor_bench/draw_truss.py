import numpy as np
import os
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sympy as sp


def evaluate_geometry(nodes, loads, subs_dict):
    """
    Translates exact symbolic coordinates and forces into raw floating-point 
    decimals for plotting with Matplotlib.
    
    Args:
        nodes (dict): The topological nodes dictionary (e.g., {'C': ('L/2', 'L')})
        loads (dict): The applied loads dictionary (e.g., {'C': (0, '-P')})
        subs_dict (dict): The numerical substitutions (e.g., {'L': 10.0, 'P': 50.0})
        
    Returns:
        tuple: (evaluated_nodes, evaluated_loads) where all values are floats.
    """
    evaluated_nodes = {}
    evaluated_loads = {}
    
    # Helper to safely digest strings, raw numbers, or SymPy objects
    def evaluate_val(val):
        expr = sp.sympify(val)
        
        # 1. Robust substitution: Match string keys to actual symbols in the expression.
        # This bypasses issues where Symbol('L', positive=True) != Symbol('L')
        for sym in getattr(expr, 'free_symbols', []):
            if sym.name in subs_dict:
                expr = expr.subs(sym, subs_dict[sym.name])
                
        # 2. Standard substitution (catches cases where the user passed actual SymPy objects as keys)
        expr = expr.subs(subs_dict)
        
        return float(expr.evalf())

    # 1. Evaluate Nodes
    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (evaluate_val(x), evaluate_val(y))
        
    # 2. Evaluate Loads
    for nid, (fx, fy) in loads.items():
        evaluated_loads[nid] = (evaluate_val(fx), evaluate_val(fy))
        
    return evaluated_nodes, evaluated_loads


def calculate_bounding_box(evaluated_nodes):
    """
    Determines the geometric extents of the truss based on evaluated nodal coordinates.
    
    Args:
        evaluated_nodes (dict): A dictionary of evaluated numerical coordinates, 
                                e.g., {'A': (0.0, 0.0), 'B': (10.0, 5.0)}
                                
    Returns:
        tuple: (min_x, max_x, min_y, max_y) as floats.
    """
    if not evaluated_nodes:
        return 0.0, 0.0, 0.0, 0.0

    # Extract all X and Y coordinates into flat lists
    x_coords = [coords[0] for coords in evaluated_nodes.values()]
    y_coords = [coords[1] for coords in evaluated_nodes.values()]
    
    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def _get_rotation_matrix(angle_degrees):
    """
    Generates a 2x2 numpy array representing a 2D counter-clockwise 
    rotation matrix for a given angle in degrees.
    """
    theta = math.radians(angle_degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    
    return np.array([
        [c, -s],
        [s,  c]
    ])


def _transform_vertices(local_vertices, global_x, global_y, angle_degrees):
    """
    Rotates local coordinates and translates them to the global node position.
    
    Args:
        local_vertices (list of tuples): e.g., [(0, 0), (-1, -1), (1, -1)]
        global_x (float): The global X coordinate of the node.
        global_y (float): The global Y coordinate of the node.
        angle_degrees (float): The orientation of the support (0 = horizontal floor).
        
    Returns:
        list of tuples: The global (X, Y) coordinates to plot.
    """
    rotation_matrix = _get_rotation_matrix(angle_degrees)
    
    global_vertices = []
    for lx, ly in local_vertices:
        # 1. Convert to numpy array for matrix multiplication
        vec = np.array([lx, ly])
        
        # 2. Rotate the vector around the local origin (0,0)
        rotated_vec = rotation_matrix.dot(vec)
        
        # 3. Translate to the global position
        gx = rotated_vec[0] + global_x
        gy = rotated_vec[1] + global_y
        
        global_vertices.append((gx, gy))
        
    return global_vertices


def _draw_ground_plane(ax, center_x, center_y, angle_degrees, scale=1.0):
    """
    Draws the textbook-style rigid ground boundary condition (a thick line 
    with short diagonal hash marks).
    """
    # 1. Define local dimensions based on the scale factor
    width = 0.8 * scale
    hash_spacing = 0.15 * scale
    
    # Hash marks project downward and to the left
    hash_dx = -0.15 * scale
    hash_dy = -0.25 * scale
    
    local_vertices = []
    
    # The primary thick ground line
    local_vertices.append((-width / 2.0, 0.0))
    local_vertices.append((width / 2.0, 0.0))
    
    # Generate the starting and ending points for each small hash mark
    num_hashes = int(width / hash_spacing) + 1
    start_x = -width / 2.0
    
    for i in range(num_hashes):
        hx = start_x + i * hash_spacing
        local_vertices.append((hx, 0.0))                 # Top of the hash
        local_vertices.append((hx + hash_dx, hash_dy))   # Bottom of the hash
        
    # 2. Apply the mathematical rotation and translation to all points at once
    global_verts = _transform_vertices(local_vertices, center_x, center_y, angle_degrees)
    
    # 3. Paint to the Matplotlib Axis
    # Draw the main line (Index 0 and 1)
    main_start, main_end = global_verts[0], global_verts[1]
    ax.plot(
        [main_start[0], main_end[0]], 
        [main_start[1], main_end[1]], 
        color='black', 
        linewidth=2.5, 
        zorder=2,
        solid_capstyle='round'
    )
    
    # Draw the hash marks (Iterate through the rest of the list in pairs)
    for i in range(2, len(global_verts), 2):
        h_start = global_verts[i]
        h_end = global_verts[i+1]
        ax.plot(
            [h_start[0], h_end[0]], 
            [h_start[1], h_end[1]], 
            color='black', 
            linewidth=1.0, 
            zorder=2,
            solid_capstyle='round'
        )


def draw_pin(ax, x, y, angle=0.0, scale=1.0):
    """
    Draws a textbook-style pin support (an equilateral triangle resting 
    on a hatched ground plane).
    """
    # 1. Define local triangle geometry (top vertex sits at the origin)
    w = 0.6 * scale
    h = w * math.sqrt(3) / 2.0
    
    local_triangle = [
        (0.0, 0.0),       # Top vertex (snaps directly to the node)
        (-w / 2.0, -h),   # Bottom left
        (w / 2.0, -h)     # Bottom right
    ]
    
    # 2. Transform the triangle to the global position and rotation
    global_triangle = _transform_vertices(local_triangle, x, y, angle)
    
    # 3. Paint the triangle onto the axis
    pin_patch = patches.Polygon(
        global_triangle, 
        closed=True, 
        facecolor='white', 
        edgecolor='black', 
        linewidth=1.5,
        zorder=3  # Above the structural lines, below the nodes
    )
    ax.add_patch(pin_patch)
    
    # 4. Find the center of the triangle's base to attach the ground plane
    # The base is at local coordinate (0, -h). We pass this single point 
    # through the transformer to find its global location.
    base_center_global = _transform_vertices([(0.0, -h)], x, y, angle)[0]
    
    # 5. Draw the ground plane exactly at the transformed base
    _draw_ground_plane(
        ax, 
        center_x=base_center_global[0], 
        center_y=base_center_global[1], 
        angle_degrees=angle, 
        scale=scale
    )


def draw_roller(ax, x, y, angle=0.0, scale=1.0):
    """
    Draws a textbook-style roller support (an equilateral triangle resting 
    on two circular rollers, sitting on a hatched ground plane).
    """
    # 1. Define base geometry variables
    w = 0.6 * scale
    h = w * math.sqrt(3) / 2.0
    r = 0.1 * scale  # Radius of the roller wheels
    
    # 2. Draw the Triangle
    local_triangle = [
        (0.0, 0.0),       # Top vertex (snaps to node)
        (-w / 2.0, -h),   # Bottom left
        (w / 2.0, -h)     # Bottom right
    ]
    global_triangle = _transform_vertices(local_triangle, x, y, angle)
    
    triangle_patch = patches.Polygon(
        global_triangle, 
        closed=True, 
        facecolor='white', 
        edgecolor='black', 
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(triangle_patch)
    
    # 3. Draw the Roller Wheels
    # The wheels sit directly below the triangle base. 
    # We space them evenly at -w/4 and +w/4.
    local_wheel_centers = [
        (-w / 4.0, -h - r),
        (w / 4.0, -h - r)
    ]
    global_wheel_centers = _transform_vertices(local_wheel_centers, x, y, angle)
    
    for center_x, center_y in global_wheel_centers:
        wheel_patch = patches.Circle(
            (center_x, center_y), 
            r, 
            facecolor='white', 
            edgecolor='black', 
            linewidth=1.5,
            zorder=3
        )
        ax.add_patch(wheel_patch)
        
    # 4. Draw the Ground Plane
    # The ground plane sits completely beneath the wheels (Y = -h - 2*r)
    local_base = [(0.0, -h - 2 * r)]
    base_center_global = _transform_vertices(local_base, x, y, angle)[0]
    
    _draw_ground_plane(
        ax, 
        center_x=base_center_global[0], 
        center_y=base_center_global[1], 
        angle_degrees=angle, 
        scale=scale
    )


def draw_elements_layer(ax, evaluated_nodes, elements, show_labels=False):
    """
    Iterates through the connectivity dictionary and draws thick structural 
    members between the evaluated coordinates.
    
    Args:
        ax: The Matplotlib axis object.
        evaluated_nodes (dict): Numerical node coordinates {'A': (x, y)}
        elements (dict): Connectivity mapping {'1': ('A', 'B')}
        show_labels (bool): If True, renders the element ID at the midpoint.
    """
    for element_id, (node_start, node_end) in elements.items():
        # Fetch the global coordinates
        x_start, y_start = evaluated_nodes[node_start]
        x_end, y_end = evaluated_nodes[node_end]
        
        # 1. Draw the structural line
        ax.plot(
            [x_start, x_end], 
            [y_start, y_end], 
            color='black',
            linewidth=4.0,
            zorder=1,
            solid_capstyle='round'
        )
        
        # 2. Optionally draw the text label
        if show_labels:
            mid_x = (x_start + x_end) / 2.0
            mid_y = (y_start + y_end) / 2.0
            
            ax.text(
                mid_x, 
                mid_y, 
                str(element_id), 
                color='black',
                fontsize=10,
                fontweight='bold',
                ha='center',       
                va='center',       
                zorder=4,          
                bbox=dict(facecolor=(0.8, 0.8, 0.8), edgecolor='none', pad=2.0)
            )


def draw_nodes_layer(ax, evaluated_nodes):
    """
    Iterates through the evaluated nodes and draws textbook-style joints 
    (white circles with dark borders) along with their text labels.
    
    Args:
        ax: The Matplotlib axis object.
        evaluated_nodes (dict): Numerical node coordinates {'A': (x, y)}
    """
    for nid, (x, y) in evaluated_nodes.items():
        # 1. Draw the physical joint
        ax.plot(
            x, y, 
            marker='o', 
            markersize=10, 
            markerfacecolor='white', 
            markeredgecolor='black', 
            markeredgewidth=2.0,
            zorder=5  # Sits above elements (1) and supports (3)
        )
        
        # 2. Draw the node label offset slightly to the top-right
        ax.annotate(
            str(nid),
            xy=(x, y),
            xytext=(8, 8),          # 8 points right, 8 points up
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='blue',
            zorder=6                # Sits above the node marker itself
        )


def draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0):
    """
    Iterates through the truss supports, dynamically calculating orientation 
    based on the truss's bounding box and cantilever detection.
    """
    min_x, max_x, min_y, max_y = calculate_bounding_box(evaluated_nodes)
    truss_center_x = (min_x + max_x) / 2.0
    truss_center_y = (min_y + max_y) / 2.0
    
    # Cantilever heuristic: Are ALL supports sitting on the exact same vertical edge?
    support_x_coords = [evaluated_nodes[n][0] for n in supports.keys()]
    is_left_cantilever = all(math.isclose(sx, min_x, abs_tol=1e-5) for sx in support_x_coords)
    is_right_cantilever = all(math.isclose(sx, max_x, abs_tol=1e-5) for sx in support_x_coords)

    for nid, constraints in supports.items():
        x, y = evaluated_nodes[nid]
        
        # PIN SUPPORT (Restricted in X and Y)
        if 'x' in constraints and 'y' in constraints:
            if is_left_cantilever:
                pin_angle = -90.0  # Mount to left wall
            elif is_right_cantilever:
                pin_angle = 90.0   # Mount to right wall
            elif math.isclose(y, max_y, abs_tol=1e-5) and not math.isclose(y, min_y, abs_tol=1e-5):
                pin_angle = 180.0  # Mount to ceiling
            else:
                pin_angle = 0.0    # Default to floor
                
            draw_pin(ax, x, y, angle=pin_angle, scale=scale_factor)
            
        # ROLLER ON A FLOOR/CEILING (Restricted in Y only)
        elif 'y' in constraints and 'x' not in constraints:
            vertical_angle = 180.0 if y > truss_center_y else 0.0
            draw_roller(ax, x, y, angle=vertical_angle, scale=scale_factor)
            
        # ROLLER ON A WALL (Restricted in X only)
        elif 'x' in constraints and 'y' not in constraints:
            horizontal_angle = 90.0 if x > truss_center_x else -90.0
            draw_roller(ax, x, y, angle=horizontal_angle, scale=scale_factor)


def draw_loads_layer(ax, evaluated_nodes, evaluated_loads, original_loads, scale_factor=1.0):
    """
    Iterates through the evaluated loads and draws standard red vector arrows 
    pushing on the nodes, prioritizing symbolic labels (e.g., 'P') over 
    numerical magnitudes.
    """
    arrow_length = 1.5 * scale_factor

    for nid, (fx, fy) in evaluated_loads.items():
        # 1. Skip if there is no numerical load
        if math.isclose(fx, 0.0, abs_tol=1e-9) and math.isclose(fy, 0.0, abs_tol=1e-9):
            continue
            
        x, y = evaluated_nodes[nid]
        
        # 2. Calculate the true vector math for drawing the arrow
        magnitude = math.hypot(fx, fy)
        dx = (fx / magnitude) * arrow_length
        dy = (fy / magnitude) * arrow_length
        
        # 3. Calculate the tail position
        tail_x = x - dx
        tail_y = y - dy
        
        # 4. Draw the vector arrow
        ax.annotate(
            "", 
            xy=(x, y),                
            xytext=(tail_x, tail_y),  
            arrowprops=dict(
                arrowstyle="-|>",     
                color='#cc0000',      
                lw=2.5,
                mutation_scale=20     
            ),
            zorder=4                  
        )
        
        # 5. Extract the symbolic label
        orig_fx, orig_fy = original_loads[nid]
        
        # Convert to strings and strip leading negatives (arrow implies direction)
        s_fx = str(orig_fx).replace(" ", "").lstrip('-')
        s_fy = str(orig_fy).replace(" ", "").lstrip('-')
        
        # Pick the active component for the label
        if s_fx != '0' and s_fx != '0.0' and s_fy != '0' and s_fy != '0.0':
            label_text = f"({s_fx}, {s_fy})" # Fallback for angled symbolic forces
        elif s_fx != '0' and s_fx != '0.0':
            label_text = s_fx
        else:
            label_text = s_fy
            
        # If the user passed a pure number instead of a variable, format it cleanly
        try:
            num_val = float(label_text)
            label_text = f"{num_val:g}"
        except ValueError:
            pass # It is a symbolic label like 'P', leave it as is
            
        # 6. Paint the label
        text_offset_x = tail_x - (dx * 0.2)
        text_offset_y = tail_y - (dy * 0.2)
        
        ax.text(
            text_offset_x, 
            text_offset_y, 
            label_text, 
            color='#cc0000', 
            fontsize=11, 
            fontweight='bold',
            ha='center', 
            va='center',
            zorder=4
        )


def generate_truss_diagram(nodes, elements, loads, supports, subs_dict, filepath, show_labels=False):
    """
    The master orchestrator. Evaluates the physical geometry and renders a 
    textbook-quality static diagram of the complete truss system.
    
    Args:
        nodes (dict): Symbolic node coordinates.
        elements (dict): Node-to-node connectivity.
        loads (dict): Applied point loads.
        supports (dict): Boundary conditions.
        subs_dict (dict): Numerical substitutions for variables.
        filepath (str): The output destination (e.g., 'truss_diagram.png').
        show_labels (bool): Whether to draw element ID labels.
    """
    # 1. Process Geometry
    evaluated_nodes, evaluated_loads = evaluate_geometry(nodes, loads, subs_dict)
    min_x, max_x, min_y, max_y = calculate_bounding_box(evaluated_nodes)
    
    # 2. Calculate Dynamic Scale Factor
    # We base the size of the supports and arrows on 10% of the maximum truss dimension
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    scale_factor = max_dim * 0.1 if max_dim > 0 else 1.0
    
    # 3. Initialize the Canvas
    # A standard 10x8 figure gives plenty of resolution for slide decks
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Strip away the default Matplotlib bounding box and axes
    
    # 4. Paint the Layers (Order matters for Z-stacking)
    draw_elements_layer(ax, evaluated_nodes, elements, show_labels=show_labels)
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=scale_factor)
    draw_nodes_layer(ax, evaluated_nodes)
    draw_loads_layer(ax, evaluated_nodes, evaluated_loads, loads, scale_factor=scale_factor)
    
    # 5. Buffer the edges
    # Adds a 20% margin around the bounding box so boundary conditions 
    # and load arrows don't get chopped off by the edge of the image.
    ax.margins(0.2)
    
    # 6. Export the File
    # dpi=300 ensures it remains crystal clear when projected on a large screen
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    
    # 7. Flush memory
    plt.close(fig)


def draw_truss_adapter(nodes, elements_list, loads, supports, metadata, subs_dict, output_dir=None):
    """
    A modular adapter that prepares generic template data for the drawing engine.
    
    Args:
        nodes (dict): The symbolic nodes dictionary.
        elements_list (list): The full elements definition.
        loads (dict): The symbolic applied loads.
        supports (dict): The boundary conditions.
        metadata (dict): Used to extract the benchmark ID for naming.
        subs_dict (dict): Exact numerical substitutions for the symbols (e.g., {'L': 10.0, 'P': 50.0}).
        output_dir (str, optional): Target directory. Defaults to a local 'draw_truss' folder 
                                    relative to the executing script's working directory.
    """
    # 1. Extract standard connectivity for the drawing engine
    drawing_elements = {elem['id']: elem['nodes'] for elem in elements_list}
    
    # 2. Dynamic Directory Routing
    if output_dir is None:
        # Fallback for local template testing
        # Uses the current working directory so images save next to the template being tested,
        # rather than inside the installed package's source directory.
        output_dir = os.path.join(os.getcwd(), "draw_truss")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Dynamic Filename
    filename = f"{metadata.get('benchmark_id', 'noID')}_truss_schematic.png"
    filepath = os.path.join(output_dir, filename)
    
    # 4. Execute the orchestrator
    generate_truss_diagram(
        nodes=nodes,
        elements=drawing_elements,
        loads=loads,
        supports=supports,
        subs_dict=subs_dict,
        filepath=filepath,
        show_labels=True
    )
    
    print(f"Schematic successfully generated and saved to: {filepath}")

