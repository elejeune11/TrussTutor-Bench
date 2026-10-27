import numpy as np
import os
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sympy as sp


def evaluate_geometry(nodes, loads, subs_dict):
    """
    Converts symbolic node coordinates and loads to floats for plotting.

    Args:
        nodes (dict): Node ID -> (x, y), e.g. {'C': ('L/2', 'L')}.
        loads (dict): Node ID -> (Fx, Fy), e.g. {'C': (0, '-P')}.
        subs_dict (dict): Numerical values for the symbols, e.g. {'L': 10.0, 'P': 50.0}.

    Returns:
        tuple: (evaluated_nodes, evaluated_loads), with all values as floats.
    """
    evaluated_nodes = {}
    evaluated_loads = {}

    def evaluate_val(val):
        expr = sp.sympify(val)

        # Match symbols by name, because Symbol('L', positive=True) != Symbol('L').
        for sym in getattr(expr, 'free_symbols', []):
            if sym.name in subs_dict:
                expr = expr.subs(sym, subs_dict[sym.name])

        # Also handles subs_dict keys that are SymPy symbols.
        expr = expr.subs(subs_dict)

        return float(expr.evalf())

    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (evaluate_val(x), evaluate_val(y))

    for nid, (fx, fy) in loads.items():
        evaluated_loads[nid] = (evaluate_val(fx), evaluate_val(fy))

    return evaluated_nodes, evaluated_loads


def calculate_bounding_box(evaluated_nodes):
    """
    Returns the extents of the truss.

    Args:
        evaluated_nodes (dict): Node ID -> (x, y) as floats, e.g. {'A': (0.0, 0.0), 'B': (10.0, 5.0)}.

    Returns:
        tuple: (min_x, max_x, min_y, max_y).
    """
    if not evaluated_nodes:
        return 0.0, 0.0, 0.0, 0.0

    x_coords = [coords[0] for coords in evaluated_nodes.values()]
    y_coords = [coords[1] for coords in evaluated_nodes.values()]

    return min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def _get_rotation_matrix(angle_degrees):
    """
    Returns the 2x2 counter-clockwise rotation matrix for an angle in degrees.

    Args:
        angle_degrees (float): Rotation angle.

    Returns:
        np.ndarray: The rotation matrix.
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
    Rotates local vertices around the origin and moves them to the node position.

    Args:
        local_vertices (list): (x, y) points, e.g. [(0, 0), (-1, -1), (1, -1)].
        global_x (float): X coordinate of the node.
        global_y (float): Y coordinate of the node.
        angle_degrees (float): Orientation of the support (0 = horizontal floor).

    Returns:
        list: The (x, y) points in global coordinates.
    """
    rotation_matrix = _get_rotation_matrix(angle_degrees)

    global_vertices = []
    for lx, ly in local_vertices:
        vec = np.array([lx, ly])
        rotated_vec = rotation_matrix.dot(vec)

        gx = rotated_vec[0] + global_x
        gy = rotated_vec[1] + global_y

        global_vertices.append((gx, gy))

    return global_vertices


def _draw_ground_plane(ax, center_x, center_y, angle_degrees, scale=1.0):
    """
    Draws the ground symbol of a support: a thick line with short diagonal hash marks.

    Args:
        ax: Matplotlib axis.
        center_x (float): X of the line's center.
        center_y (float): Y of the line's center.
        angle_degrees (float): Orientation of the support.
        scale (float): Size factor.
    """
    width = 0.8 * scale
    hash_spacing = 0.15 * scale

    # Hash marks project downward and to the left.
    hash_dx = -0.15 * scale
    hash_dy = -0.25 * scale

    local_vertices = []

    local_vertices.append((-width / 2.0, 0.0))
    local_vertices.append((width / 2.0, 0.0))

    num_hashes = int(width / hash_spacing) + 1
    start_x = -width / 2.0

    for i in range(num_hashes):
        hx = start_x + i * hash_spacing
        local_vertices.append((hx, 0.0))                 # Top of the hash
        local_vertices.append((hx + hash_dx, hash_dy))   # Bottom of the hash

    global_verts = _transform_vertices(local_vertices, center_x, center_y, angle_degrees)

    # The first two points are the main line, the rest are hash marks in pairs.
    main_start, main_end = global_verts[0], global_verts[1]
    ax.plot(
        [main_start[0], main_end[0]],
        [main_start[1], main_end[1]],
        color='black',
        linewidth=2.5,
        zorder=2,
        solid_capstyle='round'
    )

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
    Draws a pin support: a triangle on a hatched ground plane.

    Args:
        ax: Matplotlib axis.
        x (float): X of the node.
        y (float): Y of the node.
        angle (float): Orientation in degrees (0 = floor).
        scale (float): Size factor.
    """
    w = 0.6 * scale
    h = w * math.sqrt(3) / 2.0

    local_triangle = [
        (0.0, 0.0),       # Top vertex, on the node
        (-w / 2.0, -h),   # Bottom left
        (w / 2.0, -h)     # Bottom right
    ]

    global_triangle = _transform_vertices(local_triangle, x, y, angle)

    pin_patch = patches.Polygon(
        global_triangle,
        closed=True,
        facecolor='white',
        edgecolor='black',
        linewidth=1.5,
        zorder=3
    )
    ax.add_patch(pin_patch)

    base_center_global = _transform_vertices([(0.0, -h)], x, y, angle)[0]

    _draw_ground_plane(
        ax,
        center_x=base_center_global[0],
        center_y=base_center_global[1],
        angle_degrees=angle,
        scale=scale
    )


def draw_roller(ax, x, y, angle=0.0, scale=1.0):
    """
    Draws a roller support: a triangle on two wheels, on a hatched ground plane.

    Args:
        ax: Matplotlib axis.
        x (float): X of the node.
        y (float): Y of the node.
        angle (float): Orientation in degrees (0 = floor).
        scale (float): Size factor.
    """
    w = 0.6 * scale
    h = w * math.sqrt(3) / 2.0
    r = 0.1 * scale  # Wheel radius

    local_triangle = [
        (0.0, 0.0),       # Top vertex, on the node
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

    # The wheels sit under the triangle base, at -w/4 and +w/4.
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

    # The ground plane sits under the wheels.
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
    Draws the structural members as thick lines between their nodes.

    Args:
        ax: Matplotlib axis.
        evaluated_nodes (dict): Node ID -> (x, y) as floats.
        elements (dict): Element ID -> (start node, end node), e.g. {'1': ('A', 'B')}.
        show_labels (bool): Whether to write the element ID at the midpoint.
    """
    for element_id, (node_start, node_end) in elements.items():
        x_start, y_start = evaluated_nodes[node_start]
        x_end, y_end = evaluated_nodes[node_end]

        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            color='black',
            linewidth=4.0,
            zorder=1,
            solid_capstyle='round'
        )

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
    Draws the joints as white circles with a dark border, and their labels.

    Args:
        ax: Matplotlib axis.
        evaluated_nodes (dict): Node ID -> (x, y) as floats.
    """
    for nid, (x, y) in evaluated_nodes.items():
        ax.plot(
            x, y,
            marker='o',
            markersize=10,
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=2.0,
            zorder=5
        )

        # Label offset 8 points to the top-right.
        ax.annotate(
            str(nid),
            xy=(x, y),
            xytext=(8, 8),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='blue',
            zorder=6
        )


def draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0):
    """
    Draws the supports, oriented from the truss's bounding box: pins and rollers on a
    floor, ceiling or wall, with a cantilever detected when all supports share an edge.

    Args:
        ax: Matplotlib axis.
        evaluated_nodes (dict): Node ID -> (x, y) as floats.
        supports (dict): Node ID -> constrained directions, e.g. {'A': ['x', 'y']}.
        scale_factor (float): Size factor.
    """
    min_x, max_x, min_y, max_y = calculate_bounding_box(evaluated_nodes)
    truss_center_x = (min_x + max_x) / 2.0
    truss_center_y = (min_y + max_y) / 2.0

    # Cantilever: all supports sit on the same vertical edge.
    support_x_coords = [evaluated_nodes[n][0] for n in supports.keys()]
    is_left_cantilever = all(math.isclose(sx, min_x, abs_tol=1e-5) for sx in support_x_coords)
    is_right_cantilever = all(math.isclose(sx, max_x, abs_tol=1e-5) for sx in support_x_coords)

    for nid, constraints in supports.items():
        x, y = evaluated_nodes[nid]

        # Pin: restricted in x and y.
        if 'x' in constraints and 'y' in constraints:
            if is_left_cantilever:
                pin_angle = -90.0  # Left wall
            elif is_right_cantilever:
                pin_angle = 90.0   # Right wall
            elif math.isclose(y, max_y, abs_tol=1e-5) and not math.isclose(y, min_y, abs_tol=1e-5):
                pin_angle = 180.0  # Ceiling
            else:
                pin_angle = 0.0    # Floor

            draw_pin(ax, x, y, angle=pin_angle, scale=scale_factor)

        # Roller on a floor or ceiling: restricted in y only.
        elif 'y' in constraints and 'x' not in constraints:
            vertical_angle = 180.0 if y > truss_center_y else 0.0
            draw_roller(ax, x, y, angle=vertical_angle, scale=scale_factor)

        # Roller on a wall: restricted in x only.
        elif 'x' in constraints and 'y' not in constraints:
            horizontal_angle = 90.0 if x > truss_center_x else -90.0
            draw_roller(ax, x, y, angle=horizontal_angle, scale=scale_factor)


def draw_loads_layer(ax, evaluated_nodes, evaluated_loads, original_loads, scale_factor=1.0):
    """
    Draws the applied loads as red arrows pushing on the nodes. The label is the
    symbolic load (e.g. 'P') when there is one, otherwise the number.

    Args:
        ax: Matplotlib axis.
        evaluated_nodes (dict): Node ID -> (x, y) as floats.
        evaluated_loads (dict): Node ID -> (Fx, Fy) as floats.
        original_loads (dict): Node ID -> (Fx, Fy) before evaluation, used for the labels.
        scale_factor (float): Size factor.
    """
    arrow_length = 1.5 * scale_factor

    for nid, (fx, fy) in evaluated_loads.items():
        if math.isclose(fx, 0.0, abs_tol=1e-9) and math.isclose(fy, 0.0, abs_tol=1e-9):
            continue

        x, y = evaluated_nodes[nid]

        magnitude = math.hypot(fx, fy)
        dx = (fx / magnitude) * arrow_length
        dy = (fy / magnitude) * arrow_length

        tail_x = x - dx
        tail_y = y - dy

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

        orig_fx, orig_fy = original_loads[nid]

        # The arrow shows the direction, so the leading minus sign is dropped.
        s_fx = str(orig_fx).replace(" ", "").lstrip('-')
        s_fy = str(orig_fy).replace(" ", "").lstrip('-')

        if s_fx != '0' and s_fx != '0.0' and s_fy != '0' and s_fy != '0.0':
            label_text = f"({s_fx}, {s_fy})" # Angled symbolic force
        elif s_fx != '0' and s_fx != '0.0':
            label_text = s_fx
        else:
            label_text = s_fy

        try:
            num_val = float(label_text)
            label_text = f"{num_val:g}"
        except ValueError:
            pass # Symbolic label such as 'P'

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
    Draws the complete truss (members, supports, joints, loads) and saves it as an image.

    Args:
        nodes (dict): Symbolic node coordinates.
        elements (dict): Element ID -> (start node, end node).
        loads (dict): Applied point loads.
        supports (dict): Boundary conditions.
        subs_dict (dict): Numerical values for the symbols.
        filepath (str): Output file (e.g., 'truss_diagram.png').
        show_labels (bool): Whether to write the element IDs.
    """
    evaluated_nodes, evaluated_loads = evaluate_geometry(nodes, loads, subs_dict)
    min_x, max_x, min_y, max_y = calculate_bounding_box(evaluated_nodes)

    # Support size and arrow length are 10% of the largest truss dimension.
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    scale_factor = max_dim * 0.1 if max_dim > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Z-order: members 1, supports 3, loads 4, nodes 5, labels 6.
    draw_elements_layer(ax, evaluated_nodes, elements, show_labels=show_labels)
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=scale_factor)
    draw_nodes_layer(ax, evaluated_nodes)
    draw_loads_layer(ax, evaluated_nodes, evaluated_loads, loads, scale_factor=scale_factor)

    # A 20% margin keeps supports and arrows from being cut off.
    ax.margins(0.2)

    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

    plt.close(fig)


def draw_truss_adapter(nodes, elements_list, loads, supports, metadata, subs_dict, output_dir=None):
    """
    Draws a task's truss and saves it as <benchmark_id>_truss_schematic.png.

    Args:
        nodes (dict): Symbolic node coordinates.
        elements_list (list): Element dictionaries with "id" and "nodes".
        loads (dict): Symbolic applied loads.
        supports (dict): Boundary conditions.
        metadata (dict): Task metadata, used for the benchmark ID in the file name.
        subs_dict (dict): Numerical values for the symbols (e.g., {'L': 10.0, 'P': 50.0}).
        output_dir (str, optional): Output folder. Defaults to a "draw_truss" folder in the
            current working directory.
    """
    drawing_elements = {elem['id']: elem['nodes'] for elem in elements_list}

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "draw_truss")

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{metadata.get('benchmark_id', 'noID')}_truss_schematic.png"
    filepath = os.path.join(output_dir, filename)

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
