import numpy as np


def compute_element_properties(node_i, node_j):
    """
    Computes the length and direction cosines of a 2D truss element.

    Args:
        node_i (tuple): (x, y) of the start node.
        node_j (tuple): (x, y) of the end node.

    Returns:
        tuple: (L, c, s) with the length, cos(theta) and sin(theta).
    """
    dx = node_j[0] - node_i[0]
    dy = node_j[1] - node_i[1]
    L = np.sqrt(dx**2 + dy**2)

    if L == 0:
        raise ValueError("Element length cannot be zero (overlapping nodes).")

    c = dx / L
    s = dy / L
    return L, c, s


def compute_element_stiffness(E, A, L, c, s):
    """
    Computes the 4x4 element stiffness matrix in global coordinates.

    Args:
        E (float): Young's modulus.
        A (float): Cross-sectional area.
        L (float): Element length.
        c (float): cos(theta) = dx / L.
        s (float): sin(theta) = dy / L.

    Returns:
        np.ndarray: Symmetric 4x4 stiffness matrix.
    """
    if L <= 0:
        raise ValueError("Element length must be strictly positive.")

    k = (E * A) / L

    c2 = c**2
    s2 = s**2
    cs = c * s

    K_e = k * np.array([
        [ c2,  cs, -c2, -cs],
        [ cs,  s2, -cs, -s2],
        [-c2, -cs,  c2,  cs],
        [-cs, -s2,  cs,  s2]
    ], dtype=float)

    return K_e


def assemble_global_stiffness(nodes, elements):
    """
    Assembles the global stiffness matrix of the truss.

    Args:
        nodes (dict): Node ID -> (X, Y).
        elements (list): Element dictionaries with 'nodes', 'E' and 'A'.

    Returns:
        np.ndarray: The 2N x 2N global stiffness matrix.
    """
    # Map arbitrary node IDs to 0-based indices.
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    N = len(nodes)
    K_global = np.zeros((2 * N, 2 * N), dtype=float)

    for el in elements:
        n_i, n_j = el['nodes']
        E = el['E']
        A = el['A']

        L, c, s = compute_element_properties(nodes[n_i], nodes[n_j])
        K_e = compute_element_stiffness(E, A, L, c, s)

        idx_i = node_to_idx[n_i]
        idx_j = node_to_idx[n_j]

        dofs = [
            2 * idx_i,      # Node i - X displacement
            2 * idx_i + 1,  # Node i - Y displacement
            2 * idx_j,      # Node j - X displacement
            2 * idx_j + 1   # Node j - Y displacement
        ]

        K_global[np.ix_(dofs, dofs)] += K_e

    return K_global


def apply_boundary_conditions(nodes, K_global, F_global, supports):
    """
    Keeps only the free degrees of freedom of the stiffness matrix and force vector.

    Args:
        nodes (dict): Node ID -> (X, Y).
        K_global (np.ndarray): Full 2N x 2N stiffness matrix.
        F_global (np.ndarray): Full 2N force vector.
        supports (dict): Node ID -> constrained directions (e.g., ['x', 'y']).

    Returns:
        tuple: (K_reduced, F_reduced, free_dofs) with the reduced matrix, the reduced
            vector and the global indices of the free degrees of freedom.
    """
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    total_dofs = 2 * len(nodes)
    all_dofs = set(range(total_dofs))
    constrained_dofs = set()

    for node_id, constraints in supports.items():
        if node_id not in node_to_idx:
            continue
        idx = node_to_idx[node_id]
        if 'x' in constraints:
            constrained_dofs.add(2 * idx)
        if 'y' in constraints:
            constrained_dofs.add(2 * idx + 1)

    free_dofs = sorted(list(all_dofs - constrained_dofs))

    if not free_dofs:
        raise ValueError("The structure is completely constrained; no free DOFs remain.")

    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_global[free_dofs]

    return K_reduced, F_reduced, free_dofs


def compute_internal_forces(nodes, elements, U_global):
    """
    Computes the axial force in every element.

    Args:
        nodes (dict): Node ID -> (X, Y).
        elements (list): Element dictionaries.
        U_global (np.ndarray): Full 2N displacement vector.

    Returns:
        dict: Element ID -> axial force (positive = tension, negative = compression).
    """
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    internal_forces = {}

    for el in elements:
        el_id = el['id']
        n_i, n_j = el['nodes']
        E, A = el['E'], el['A']

        L, c, s = compute_element_properties(nodes[n_i], nodes[n_j])

        idx_i = node_to_idx[n_i]
        idx_j = node_to_idx[n_j]

        u_ix = U_global[2 * idx_i]
        u_iy = U_global[2 * idx_i + 1]
        u_jx = U_global[2 * idx_j]
        u_jy = U_global[2 * idx_j + 1]

        # F = (EA/L) * [-c, -s, c, s] dot [u_ix, u_iy, u_jx, u_jy]^T
        deformation = (-c * u_ix) + (-s * u_iy) + (c * u_jx) + (s * u_jy)
        force = (E * A / L) * deformation

        if abs(force) < 1e-9:
            force = 0.0

        internal_forces[el_id] = force

    return internal_forces


def assemble_force_vector(nodes, loads):
    """
    Assembles the global force vector from the applied loads.

    Args:
        nodes (dict): Node ID -> (X, Y).
        loads (dict): Node ID -> (Fx, Fy).

    Returns:
        np.ndarray: The 2N force vector.
    """
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    F_global = np.zeros(2 * len(nodes), dtype=float)

    for node_id, (Fx, Fy) in loads.items():
        if node_id in node_to_idx:
            idx = node_to_idx[node_id]
            F_global[2 * idx] = Fx
            F_global[2 * idx + 1] = Fy

    return F_global


def solve_truss(nodes, elements, supports, loads):
    """
    Solves the truss with the direct stiffness method.

    Args:
        nodes (dict): Node ID -> (X, Y).
        elements (list): Element dictionaries with 'id', 'nodes', 'E' and 'A'.
        supports (dict): Node ID -> constrained directions (e.g., ['x', 'y']).
        loads (dict): Node ID -> (Fx, Fy).

    Returns:
        tuple: (U_global, internal_forces, reactions) with the displacement vector, the
            member forces (positive = tension) and the reaction vector.
    """
    # Convert to floats so symbolic inputs do not break the numerical solve.
    nodes_float = {nid: (float(x), float(y)) for nid, (x, y) in nodes.items()}

    elements_float = []
    for el in elements:
        new_el = el.copy()
        new_el['E'] = float(el['E'])
        new_el['A'] = float(el['A'])
        elements_float.append(new_el)

    loads_float = {nid: (float(fx), float(fy)) for nid, (fx, fy) in loads.items()}

    K_global = assemble_global_stiffness(nodes_float, elements_float)
    F_global = assemble_force_vector(nodes_float, loads_float)

    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(nodes_float, K_global, F_global, supports)

    U_reduced = np.linalg.solve(K_reduced, F_reduced)

    U_global = np.zeros(2 * len(nodes), dtype=float)
    U_global[free_dofs] = U_reduced

    internal_forces = compute_internal_forces(nodes_float, elements_float, U_global)

    # Reactions = (K_global * U_global) - F_applied
    reactions = (K_global @ U_global) - F_global

    # Remove floating point noise.
    reactions[np.abs(reactions) < 1e-9] = 0.0
    U_global[np.abs(U_global) < 1e-12] = 0.0

    return U_global, internal_forces, reactions


def check_stability_and_determinacy(nodes, elements, supports):
    """
    Checks the static determinacy and geometric stability of the truss.

    Args:
        nodes (dict): Node ID -> (X, Y).
        elements (list): Element dictionaries.
        supports (dict): Node ID -> constrained directions.

    Returns:
        dict: Member count "m", reaction count "r", joint count "j", "solvable_by_hand",
            "is_singular" (the stiffness matrix is singular) and a "status" text.
    """
    j = len(nodes)
    m = len(elements)

    r = sum(len(constraints) for constraints in supports.values())

    # Maxwell's rule: 2j = m + r
    is_determinate = False
    status = ""

    if m + r < 2 * j:
        status = "Unstable (Deficient members or supports)"
    elif m + r == 2 * j:
        status = "Statically Determinate"
        is_determinate = True
    else:
        degree_of_indeterminacy = (m + r) - 2 * j
        status = f"Statically Indeterminate (Degree {degree_of_indeterminacy})"

    K_global = assemble_global_stiffness(nodes, elements)
    F_dummy = np.zeros(2 * j) # Dummy load, only needed to call the partitioner.

    try:
        K_reduced, _, free_dofs = apply_boundary_conditions(nodes, K_global, F_dummy, supports)
    except ValueError:
        return {
            "m": m, "r": r, "j": j,
            "solvable_by_hand": False,
            "is_singular": True,
            "status": "Unstable (Boundary Condition Error)"
        }

    # A stable structure has a full-rank reduced matrix; lower rank means a mechanism.
    expected_rank = len(free_dofs)
    actual_rank = np.linalg.matrix_rank(K_reduced)
    is_singular = bool(actual_rank < expected_rank)

    if is_singular:
        status += " BUT Geometrically Unstable (Mechanism)"
        is_determinate = False

    return {
        "m": m,
        "r": r,
        "j": j,
        "solvable_by_hand": is_determinate and not is_singular,
        "is_singular": is_singular,
        "status": status
    }
