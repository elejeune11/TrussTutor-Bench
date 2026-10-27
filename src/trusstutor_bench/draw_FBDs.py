"""
Draws textbook-style free body diagrams: whole truss, single joint, and a method-of-sections cut.
"""

import math
import sympy as sp
import matplotlib.pyplot as plt

from trusstutor_bench.draw_truss import (
    evaluate_geometry,
    calculate_bounding_box,
    draw_elements_layer,
    draw_nodes_layer
)


def draw_fbd_node(ax, node_id: str, x: float, y: float):
    """
    Draws a joint as a white circle with the node ID centered inside it.

    Args:
        ax: Matplotlib axis.
        node_id (str): Node label (e.g., 'A').
        x (float): X coordinate.
        y (float): Y coordinate.
    """
    ax.plot(
        x, y,
        marker='o',
        markersize=24,
        markerfacecolor='white',
        markeredgecolor='black',
        markeredgewidth=2.0,
        zorder=5
    )

    ax.text(
        x, y,
        str(node_id),
        fontsize=12,
        fontweight='bold',
        color='blue',
        ha='center',
        va='center',
        zorder=6
    )


def draw_force_vector(ax, tail_x: float, tail_y: float, dx: float, dy: float, label: str, color: str = '#cc0000', label_loc: str = 'head'):
    """
    Draws a force arrow with a label near its head or tail. Zero-length vectors are skipped.

    Args:
        ax: Matplotlib axis.
        tail_x (float): X of the arrow's tail.
        tail_y (float): Y of the arrow's tail.
        dx (float): X component of the force.
        dy (float): Y component of the force.
        label (str): Label text (e.g., '$F_{AB}$ = 50 kN').
        color (str): Color of the arrow and text.
        label_loc (str): 'head' or 'tail', where the label is placed.
    """
    magnitude = math.hypot(dx, dy)
    if magnitude < 1e-9:
        return

    head_x = tail_x + dx
    head_y = tail_y + dy

    ax.annotate(
        "",
        xy=(head_x, head_y),
        xytext=(tail_x, tail_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2.5,
            mutation_scale=20
        ),
        zorder=4
    )

    offset_distance = 0.05
    unit_dx = dx / magnitude
    unit_dy = dy / magnitude

    if label_loc == 'tail':
        # Place the label behind the tail, growing outward.
        text_x = tail_x - (unit_dx * offset_distance)
        text_y = tail_y - (unit_dy * offset_distance)

        ha = 'right' if unit_dx > 0.1 else ('left' if unit_dx < -0.1 else 'center')
        va = 'top' if unit_dy > 0.1 else ('bottom' if unit_dy < -0.1 else 'center')
    else:
        # Place the label beyond the arrowhead.
        text_x = head_x + (unit_dx * offset_distance)
        text_y = head_y + (unit_dy * offset_distance)

        ha = 'left' if unit_dx > 0.1 else ('right' if unit_dx < -0.1 else 'center')
        va = 'bottom' if unit_dy > 0.1 else ('top' if unit_dy < -0.1 else 'center')

    ax.text(
        text_x,
        text_y,
        label,
        color=color,
        fontsize=12,
        fontweight='bold',
        ha=ha,
        va=va,
        zorder=4,
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.8)
    )


def draw_cut_stub(ax, start_x: float, start_y: float, unit_dx: float, unit_dy: float, stub_length: float = 1.0):
    """
    Draws a short thick line for a severed member, pointing away from the joint.

    Args:
        ax: Matplotlib axis.
        start_x (float): X of the joint.
        start_y (float): Y of the joint.
        unit_dx (float): X of the unit vector pointing away from the joint.
        unit_dy (float): Y of the unit vector pointing away from the joint.
        stub_length (float): Length of the stub.
    """
    end_x = start_x + (unit_dx * stub_length)
    end_y = start_y + (unit_dy * stub_length)

    ax.plot(
        [start_x, end_x],
        [start_y, end_y],
        color='black',
        linewidth=4.0,
        zorder=1,
        solid_capstyle='butt'  # Flat tip, so the member looks severed.
    )


def draw_intact_element(ax, x1: float, y1: float, x2: float, y2: float):
    """
    Draws a full-length member between two points.

    Args:
        ax: Matplotlib axis.
        x1 (float): X of the start.
        y1 (float): Y of the start.
        x2 (float): X of the end.
        y2 (float): Y of the end.
    """
    ax.plot(
        [x1, x2],
        [y1, y2],
        color='black',
        linewidth=4.0,
        zorder=1,
        solid_capstyle='round'
    )


def safe_float(raw_val, subs_dict: dict = None) -> float:
    """
    Substitutes known values and sets any remaining symbols to 1.0, only for drawing.

    Args:
        raw_val: Number, string or SymPy expression.
        subs_dict (dict, optional): Values for the known symbols (e.g., P or m).

    Returns:
        float: The numeric value.
    """
    subs_dict = subs_dict or {}
    expr = sp.sympify(raw_val)
    expr = expr.subs(subs_dict)
    for sym in getattr(expr, 'free_symbols', []):
        expr = expr.subs(sym, 1.0)
    return float(expr.evalf())


def aggregate_node_forces(state, node_id: str, subs_dict: dict = None) -> dict:
    """
    Collects the forces acting on one node: external loads, support reactions and
    the members meeting there (known or unknown).

    Args:
        state (TrussState): The truss state.
        node_id (str): Node to inspect.
        subs_dict (dict, optional): Values for the symbols.

    Returns:
        dict: Lists "external_loads", "support_reactions" and "internal_forces".
    """
    subs_dict = subs_dict or {}
    aggregated = {
        "external_loads": [],
        "support_reactions": [],
        "internal_forces": []
    }

    if hasattr(state, 'loads') and node_id in state.loads:
        fx, fy = state.loads[node_id]
        aggregated["external_loads"].append({"fx": safe_float(fx, subs_dict), "fy": safe_float(fy, subs_dict)})

    if hasattr(state, 'global_reactions'):
        rx_key = f"R_{node_id}_x"
        ry_key = f"R_{node_id}_y"

        if rx_key in state.global_reactions:
            aggregated["support_reactions"].append({
                "id": rx_key,
                "magnitude": safe_float(state.global_reactions[rx_key], subs_dict),
                "axis": "x"
            })

        if ry_key in state.global_reactions:
            aggregated["support_reactions"].append({
                "id": ry_key,
                "magnitude": safe_float(state.global_reactions[ry_key], subs_dict),
                "axis": "y"
            })

    if hasattr(state, 'elements'):
        for element in state.elements:
            elem_id = element['id']
            n1, n2 = element['nodes']

            if n1 == node_id or n2 == node_id:
                other_node = n2 if n1 == node_id else n1

                if hasattr(state, 'known_forces') and elem_id in state.known_forces:
                    status = "known"
                    magnitude = safe_float(state.known_forces[elem_id], subs_dict)
                else:
                    status = "unknown"
                    magnitude = None

                aggregated["internal_forces"].append({
                    "element_id": elem_id,
                    "connected_node": other_node,
                    "status": status,
                    "magnitude": magnitude
                })

    return aggregated


def generate_global_fbd(state, subs_dict: dict, filepath: str):
    """
    Draws the whole truss with each support replaced by its reaction forces, and saves it.

    Args:
        state (TrussState): Solved state with reactions.
        subs_dict (dict): Numerical values for the symbols.
        filepath (str): Output image file.
    """
    loads_dict = getattr(state, 'loads', {})
    evaluated_nodes, evaluated_loads = evaluate_geometry(
        nodes=state.nodes,
        loads=loads_dict,
        subs_dict=subs_dict
    )

    min_x, max_x, min_y, max_y = calculate_bounding_box(evaluated_nodes)

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)
    scale_factor = max_dim * 0.15 if max_dim > 0 else 1.0
    arrow_length = 1.5 * scale_factor

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    elements_dict = {elem['id']: elem['nodes'] for elem in state.elements}
    draw_elements_layer(ax, evaluated_nodes, elements_dict, show_labels=True)

    for nid, (efx, efy) in evaluated_loads.items():
        if math.isclose(efx, 0.0, abs_tol=1e-9) and math.isclose(efy, 0.0, abs_tol=1e-9):
            continue

        mag = math.hypot(efx, efy)
        dx = (efx / mag) * arrow_length
        dy = (efy / mag) * arrow_length
        nx, ny = evaluated_nodes[nid]

        tail_x = nx - dx
        tail_y = ny - dy

        draw_force_vector(ax, tail_x, tail_y, dx, dy, label=f"{mag:g}", color="#cc0000", label_loc="tail")

    draw_nodes_layer(ax, evaluated_nodes)

    if hasattr(state, 'global_reactions'):
        def eval_reaction(raw_val):
            expr = sp.sympify(raw_val)
            for sym in getattr(expr, 'free_symbols', []):
                if sym.name in subs_dict:
                    expr = expr.subs(sym, subs_dict[sym.name])
            return float(expr.subs(subs_dict).evalf())

        for nid, coords in evaluated_nodes.items():
            nx, ny = coords
            rx_key = f"R_{nid}_x"
            ry_key = f"R_{nid}_y"

            if rx_key in state.global_reactions:
                val = eval_reaction(state.global_reactions[rx_key])
                direction = 1.0 if val >= -1e-9 else -1.0
                dx = direction * arrow_length

                tail_x = nx - dx
                tail_y = ny
                label = f"$R_{{{nid}x}}$ = {abs(val):.2f}"

                draw_force_vector(ax, tail_x, tail_y, dx, 0.0, label, color="purple", label_loc="tail")

            if ry_key in state.global_reactions:
                val = eval_reaction(state.global_reactions[ry_key])
                direction = 1.0 if val >= -1e-9 else -1.0
                dy = direction * arrow_length

                tail_x = nx
                tail_y = ny - dy
                label = f"$R_{{{nid}y}}$ = {abs(val):.2f}"

                draw_force_vector(ax, tail_x, tail_y, 0.0, dy, label, color="purple", label_loc="tail")

    ax.margins(0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_node_fbd(state, target_node_id: str, subs_dict: dict, filepath: str):
    """
    Draws one isolated joint with its member stubs and all forces acting on it, and saves it.

    Args:
        state (TrussState): The truss state.
        target_node_id (str): Joint to draw.
        subs_dict (dict): Numerical values for the symbols.
        filepath (str): Output image file.
    """
    loads_dict = getattr(state, 'loads', {})
    evaluated_nodes, evaluated_loads = evaluate_geometry(
        nodes=state.nodes,
        loads=loads_dict,
        subs_dict=subs_dict
    )

    forces = aggregate_node_forces(state, target_node_id, subs_dict)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    nx, ny = evaluated_nodes[target_node_id]

    # Fixed scale for the isolated view.
    scale = 1.0
    stub_len = 1.0 * scale
    arrow_len = 1.5 * scale
    node_margin = 0.4 * scale  # Keeps incoming arrows off the node marker.

    for force in forces["internal_forces"]:
        cx, cy = evaluated_nodes[force["connected_node"]]
        dx = cx - nx
        dy = cy - ny
        dist = math.hypot(dx, dy)

        # Unit vector pointing away from the target node.
        ux, uy = dx / dist, dy / dist

        draw_cut_stub(ax, nx, ny, ux, uy, stub_length=stub_len)

        # Forces point away from the joint (tension assumed), so a negative value means compression.
        # Known forces are blue with a value; unknown forces are black, labelled only.
        is_known = force["status"] == "known"
        if is_known:
            val = float(force["magnitude"])
            label = f"$F_{{{target_node_id}{force['connected_node']}}}$ = {val:.2f}"
            color = "blue"
        else:
            label = f"$F_{{{target_node_id}{force['connected_node']}}}$"
            color = "black"

        tail_x = nx + (ux * stub_len)
        tail_y = ny + (uy * stub_len)
        vec_dx = ux * arrow_len
        vec_dy = uy * arrow_len

        draw_force_vector(ax, tail_x, tail_y, vec_dx, vec_dy, label, color=color)

    for load in forces["external_loads"]:
        fx, fy = load["fx"], load["fy"]
        if math.isclose(fx, 0.0, abs_tol=1e-9) and math.isclose(fy, 0.0, abs_tol=1e-9):
            continue

        mag = math.hypot(fx, fy)
        ux, uy = fx / mag, fy / mag

        head_x = nx - (ux * node_margin)
        head_y = ny - (uy * node_margin)

        vdx = ux * arrow_len
        vdy = uy * arrow_len
        tail_x = head_x - vdx
        tail_y = head_y - vdy

        draw_force_vector(ax, tail_x, tail_y, vdx, vdy, label=f"{mag:g}", color="#cc0000", label_loc="tail")

    for rx in forces["support_reactions"]:
        val = rx["magnitude"]
        axis = rx["axis"]
        direction = 1.0 if val >= -1e-9 else -1.0

        if axis == 'x':
            ux, uy = direction, 0.0
        else:
            ux, uy = 0.0, direction

        head_x = nx - (ux * node_margin)
        head_y = ny - (uy * node_margin)

        vdx = ux * arrow_len
        vdy = uy * arrow_len
        tail_x = head_x - vdx
        tail_y = head_y - vdy

        label = f"$R_{{{target_node_id}{axis}}}$ = {abs(val):.2f}"
        draw_force_vector(ax, tail_x, tail_y, vdx, vdy, label, color="purple", label_loc="tail")

    draw_fbd_node(ax, target_node_id, nx, ny)

    # Fixed view around the joint.
    ax.set_xlim(nx - (3.5 * scale), nx + (3.5 * scale))
    ax.set_ylim(ny - (3.5 * scale), ny + (3.5 * scale))

    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def generate_section_fbd(state, cut_element_ids: list, target_fbd_node: str, subs_dict: dict, filepath: str):
    """
    Draws the free body diagram of one side of a method-of-sections cut, and saves it.
    The kept side is the part of the truss connected to the target node once the cut
    members are removed.

    Args:
        state (TrussState): The truss state.
        cut_element_ids (list): IDs of the members the cut passes through.
        target_fbd_node (str): A node on the side to draw.
        subs_dict (dict): Numerical values for the symbols.
        filepath (str): Output image file.
    """
    loads_dict = getattr(state, 'loads', {})
    evaluated_nodes, evaluated_loads = evaluate_geometry(
        nodes=state.nodes,
        loads=loads_dict,
        subs_dict=subs_dict
    )

    # Breadth-first search from the target node, ignoring the cut members.
    adj = {n: [] for n in state.nodes}
    for elem in state.elements:
        if elem['id'] not in cut_element_ids:
            n1, n2 = elem['nodes']
            adj[n1].append(n2)
            adj[n2].append(n1)

    kept_nodes = set([target_fbd_node])
    queue = [target_fbd_node]
    while queue:
        curr = queue.pop(0)
        for neighbor in adj[curr]:
            if neighbor not in kept_nodes:
                kept_nodes.add(neighbor)
                queue.append(neighbor)

    kept_coords = {n: evaluated_nodes[n] for n in kept_nodes}
    min_x, max_x, min_y, max_y = calculate_bounding_box(kept_coords)

    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height)

    # If the kept side is a single point, scale from the whole truss.
    if max_dim < 1e-5:
        f_min_x, f_max_x, f_min_y, f_max_y = calculate_bounding_box(evaluated_nodes)
        max_dim = max(f_max_x - f_min_x, f_max_y - f_min_y)

    scale = max_dim * 0.2 if max_dim > 1e-5 else 1.0
    stub_len = 0.5 * scale
    arrow_len = 1.0 * scale
    node_margin = 0.3 * scale

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    for elem in state.elements:
        n1, n2 = elem['nodes']
        if n1 in kept_nodes and n2 in kept_nodes:
            x1, y1 = evaluated_nodes[n1]
            x2, y2 = evaluated_nodes[n2]
            draw_intact_element(ax, x1, y1, x2, y2)

    for elem in state.elements:
        eid = elem['id']
        n1, n2 = elem['nodes']

        # Exactly one end of the member is inside the kept section.
        if (n1 in kept_nodes) != (n2 in kept_nodes):
            keep_n = n1 if n1 in kept_nodes else n2
            drop_n = n2 if n1 in kept_nodes else n1

            kx, ky = evaluated_nodes[keep_n]
            dx, dy = evaluated_nodes[drop_n]

            # Unit vector pointing away from the kept section.
            vec_x, vec_y = dx - kx, dy - ky
            dist = math.hypot(vec_x, vec_y)
            ux, uy = vec_x / dist, vec_y / dist

            draw_cut_stub(ax, kx, ky, ux, uy, stub_length=stub_len)

            # Forces point away from the section (tension assumed), so a negative value means compression.
            is_known = hasattr(state, 'known_forces') and eid in state.known_forces
            if is_known:
                val = safe_float(state.known_forces[eid], subs_dict)
                label = f"$F_{{{eid}}}$ = {val:.2f}"
                color = "blue"
            else:
                label = f"$F_{{{eid}}}$"
                color = "black"

            tail_x = kx + (ux * stub_len)
            tail_y = ky + (uy * stub_len)
            vdx = ux * arrow_len
            vdy = uy * arrow_len

            draw_force_vector(ax, tail_x, tail_y, vdx, vdy, label, color=color)

    for kn in kept_nodes:
        forces = aggregate_node_forces(state, kn, subs_dict)
        nx, ny = evaluated_nodes[kn]

        for load in forces["external_loads"]:
            fx, fy = load["fx"], load["fy"]
            if math.isclose(fx, 0.0, abs_tol=1e-9) and math.isclose(fy, 0.0, abs_tol=1e-9):
                continue
            mag = math.hypot(fx, fy)
            ux, uy = fx / mag, fy / mag

            head_x, head_y = nx - (ux * node_margin), ny - (uy * node_margin)
            vdx, vdy = ux * arrow_len, uy * arrow_len
            tail_x, tail_y = head_x - vdx, head_y - vdy

            draw_force_vector(ax, tail_x, tail_y, vdx, vdy, label=f"{mag:g}", color="#cc0000", label_loc="tail")

        for rx in forces["support_reactions"]:
            val, axis = rx["magnitude"], rx["axis"]
            direction = 1.0 if val >= -1e-9 else -1.0
            ux, uy = (direction, 0.0) if axis == 'x' else (0.0, direction)

            head_x, head_y = nx - (ux * node_margin), ny - (uy * node_margin)
            vdx, vdy = ux * arrow_len, uy * arrow_len
            tail_x, tail_y = head_x - vdx, head_y - vdy

            label = f"$R_{{{kn}{axis}}}$ = {abs(val):.2f}"
            draw_force_vector(ax, tail_x, tail_y, vdx, vdy, label, color="purple", label_loc="tail")

    for kn in kept_nodes:
        nx, ny = evaluated_nodes[kn]
        draw_fbd_node(ax, kn, nx, ny)

    padding = max_dim * 0.5 if max_dim > 1e-5 else 2.0
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
