import numpy as np


def main():
    print("Hello from trusstutor-support-code!")


def compute_element_properties(node_i, node_j):
    """
    Computes length and direction cosines for a 2D truss element.
    
    Parameters:
        node_i (tuple): (x, y) coordinates of start node
        node_j (tuple): (x, y) coordinates of end node
        
    Returns:
        L (float): Element length
        c (float): cos(theta)
        s (float): sin(theta)
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
    Computes the 4x4 element stiffness matrix in the global coordinate system.

    Parameters:
        E (float): Young's Modulus
        A (float): Cross-sectional area
        L (float): Element length
        c (float): Direction cosine (cos(theta) = dx / L)
        s (float): Direction sine (sin(theta) = dy / L)
        
    Returns:
        np.ndarray: A 4x4 symmetric global stiffness matrix for the element.
    """
    if L <= 0:
        raise ValueError("Element length must be strictly positive.")

    # Constant axial stiffness term
    k = (E * A) / L

    # Precompute squared and cross terms to minimize CPU operations
    c2 = c**2
    s2 = s**2
    cs = c * s

    # Assemble the explicit 4x4 matrix
    K_e = k * np.array([
        [ c2,  cs, -c2, -cs],
        [ cs,  s2, -cs, -s2],
        [-c2, -cs,  c2,  cs],
        [-cs, -s2,  cs,  s2]
    ], dtype=float)

    return K_e


def assemble_global_stiffness(nodes, elements):
    """
    Assembles the global stiffness matrix for the entire truss system.
    
    Parameters:
        nodes (dict): Mapping of node_id -> (X, Y) coordinates.
        elements (list): List of element dictionaries containing 'nodes', 'E', and 'A'.
        
    Returns:
        np.ndarray: The unassembled 2N x 2N global stiffness matrix.
    """
    # 1. Create a stable mapping from arbitrary Node IDs to 0-based array indices
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    N = len(nodes)
    K_global = np.zeros((2 * N, 2 * N), dtype=float)
    
    for el in elements:
        n_i, n_j = el['nodes']
        E = el['E']
        A = el['A']
        
        # 2. Get local properties and local stiffness matrix
        L, c, s = compute_element_properties(nodes[n_i], nodes[n_j])
        K_e = compute_element_stiffness(E, A, L, c, s)
        
        # 3. Calculate Global Degrees of Freedom (DOFs)
        idx_i = node_to_idx[n_i]
        idx_j = node_to_idx[n_j]
        
        dofs = [
            2 * idx_i,      # Node i - X displacement
            2 * idx_i + 1,  # Node i - Y displacement
            2 * idx_j,      # Node j - X displacement
            2 * idx_j + 1   # Node j - Y displacement
        ]
        
        # 4. Scatter add the 4x4 local matrix into the 2Nx2N global matrix
        K_global[np.ix_(dofs, dofs)] += K_e
        
    return K_global


def apply_boundary_conditions(nodes, K_global, F_global, supports):
    """
    Partitions the global stiffness matrix and force vector to isolate 
    the free degrees of freedom.
    
    Parameters:
        nodes (dict): Mapping of node_id -> (X, Y) coordinates.
        K_global (np.ndarray): Full 2N x 2N global stiffness matrix.
        F_global (np.ndarray): Full 2N global force vector.
        supports (dict): Mapping of node_id -> list of constrained directions (e.g., ['x', 'y']).
        
    Returns:
        K_reduced (np.ndarray): Submatrix for free DOFs.
        F_reduced (np.ndarray): Subvector for free DOFs.
        free_dofs (list): Global indices of the free degrees of freedom.
    """
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    total_dofs = 2 * len(nodes)
    all_dofs = set(range(total_dofs))
    constrained_dofs = set()
    
    # Identify which global DOF indices are restricted by supports
    for node_id, constraints in supports.items():
        if node_id not in node_to_idx:
            continue
        idx = node_to_idx[node_id]
        if 'x' in constraints:
            constrained_dofs.add(2 * idx)
        if 'y' in constraints:
            constrained_dofs.add(2 * idx + 1)
            
    # Free DOFs are everything else
    free_dofs = sorted(list(all_dofs - constrained_dofs))
    
    if not free_dofs:
        raise ValueError("The structure is completely constrained; no free DOFs remain.")
        
    # Partition the matrix and vector using NumPy advanced slicing
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_global[free_dofs]
    
    return K_reduced, F_reduced, free_dofs


def compute_internal_forces(nodes, elements, U_global):
    """
    Computes the internal axial force for every element in the truss.
    
    Parameters:
        nodes (dict): Mapping of node_id -> (X, Y) coordinates.
        elements (list): List of element dictionaries.
        U_global (np.ndarray): Full 2N global displacement vector.
        
    Returns:
        dict: Mapping of element_id -> axial force (Positive = Tension, Negative = Compression).
    """
    node_ids = list(nodes.keys())
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
    
    internal_forces = {}
    
    for el in elements:
        el_id = el['id']
        n_i, n_j = el['nodes']
        E, A = el['E'], el['A']
        
        # 1. Get geometric properties
        L, c, s = compute_element_properties(nodes[n_i], nodes[n_j])
        
        # 2. Extract global DOFs for this specific element
        idx_i = node_to_idx[n_i]
        idx_j = node_to_idx[n_j]
        
        u_ix = U_global[2 * idx_i]
        u_iy = U_global[2 * idx_i + 1]
        u_jx = U_global[2 * idx_j]
        u_jy = U_global[2 * idx_j + 1]
        
        # 3. Calculate internal force
        # F = (EA/L) * [-c, -s, c, s] dot [u_ix, u_iy, u_jx, u_jy]^T
        deformation = (-c * u_ix) + (-s * u_iy) + (c * u_jx) + (s * u_jy)
        force = (E * A / L) * deformation
        
        # Round extremely small floating point artifacts to zero
        if abs(force) < 1e-9:
            force = 0.0
            
        internal_forces[el_id] = force
        
    return internal_forces


def assemble_force_vector(nodes, loads):
    """
    Assembles the 2N global force vector from applied external loads.
    
    Parameters:
        nodes (dict): Mapping of node_id -> (X, Y) coordinates.
        loads (dict): Mapping of node_id -> (Fx, Fy) applied forces.
        
    Returns:
        np.ndarray: The 2N global force vector.
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
    # 0. Coerce inputs to floats (The "Sanitization Layer")
    nodes_float = {nid: (float(x), float(y)) for nid, (x, y) in nodes.items()}
    
    elements_float = []
    for el in elements:
        new_el = el.copy()
        new_el['E'] = float(el['E'])
        new_el['A'] = float(el['A'])
        elements_float.append(new_el)
        
    loads_float = {nid: (float(fx), float(fy)) for nid, (fx, fy) in loads.items()}

    # 1. Build the global system (USING THE _FLOAT VERSIONS)
    K_global = assemble_global_stiffness(nodes_float, elements_float)
    F_global = assemble_force_vector(nodes_float, loads_float)
    
    # 2. Partition matrices (USING THE _FLOAT VERSIONS)
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(nodes_float, K_global, F_global, supports)
    
    # 3. Solve for unknown free displacements
    U_reduced = np.linalg.solve(K_reduced, F_reduced)
    
    # 4. Reconstruct the full 2N displacement vector
    U_global = np.zeros(2 * len(nodes), dtype=float)
    U_global[free_dofs] = U_reduced
    
    # 5. Post-process to find internal forces and support reactions
    # Use float versions here to ensure compute_internal_forces doesn't crash on symbolic geometry
    internal_forces = compute_internal_forces(nodes_float, elements_float, U_global)
    
    # 6. Reactions = (K_global * U_global) - F_applied
    reactions = (K_global @ U_global) - F_global
    
    # 7. Clean up floating point noise
    reactions[np.abs(reactions) < 1e-9] = 0.0
    U_global[np.abs(U_global) < 1e-12] = 0.0
    
    return U_global, internal_forces, reactions


def check_stability_and_determinacy(nodes, elements, supports):
    """
    Evaluates the static determinacy and geometric stability of the truss system.
    
    Returns a dictionary containing the diagnostic parameters, whether the 
    problem can be solved via basic statics, and whether the stiffness matrix is singular.
    """
    j = len(nodes)
    m = len(elements)
    
    # Calculate total number of reaction force unknowns (r)
    r = sum(len(constraints) for constraints in supports.values())
    
    # 1. Theoretical Counting (Maxwell's Rule: 2j = m + r)
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
        
    # 2. Geometric Stability Check (Matrix Singularity)
    # Assemble the matrix to see if the geometry creates a mechanism
    K_global = assemble_global_stiffness(nodes, elements)
    F_dummy = np.zeros(2 * j) # Dummy load just to pass through the partitioner
    
    try:
        K_reduced, _, free_dofs = apply_boundary_conditions(nodes, K_global, F_dummy, supports)
    except ValueError:
        return {
            "m": m, "r": r, "j": j,
            "solvable_by_hand": False,
            "is_singular": True,
            "status": "Unstable (Boundary Condition Error)"
        }

    # A stable structure's reduced matrix must have full rank. 
    # If rank < number of free DOFs, it is singular (a geometric mechanism).
    expected_rank = len(free_dofs)
    actual_rank = np.linalg.matrix_rank(K_reduced)
    is_singular = is_singular = bool(actual_rank < expected_rank)
    
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


if __name__ == "__main__":
    main()