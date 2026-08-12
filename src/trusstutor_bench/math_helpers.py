import sympy as sp


def get_unit_vector(node_id, element, nodes):
    """
    Calculates the direction cosines (c, s) for a member, pointing STRICTLY 
    away from the evaluated node (enforcing the assumption of Tension).
    """
    n1, n2 = element['nodes']
    # Identify which node is the "other" end of the member
    target_node = n2 if n1 == node_id else n1
    
    x1, y1 = nodes[node_id]
    x2, y2 = nodes[target_node]
    
    # 1. Simplify the differences immediately
    dx = sp.simplify(x2 - x1)
    dy = sp.simplify(y2 - y1)
    
    # 2. Simplify the sum of squares BEFORE taking the square root!
    # This forces SymPy to apply trig identities (cos^2 + sin^2 = 1) 
    # so the length reduces cleanly to 'L' instead of a messy radical.
    length_squared = sp.simplify(dx**2 + dy**2)
    length = sp.sqrt(length_squared)
    
    # 3. Simplify the final unit vector components
    c = sp.simplify(dx / length)
    s = sp.simplify(dy / length)
    
    return c, s


def are_collinear(node_id, el1, el2, nodes):
    """
    Determines if two elements are collinear at a given node using 
    exact 2D cross-products of raw coordinate differences to avoid 
    floating-point noise from unit vector magnitude calculations.
    """
    # 1. Identify the 'far' node for each element
    n1 = el1['nodes'][0] if el1['nodes'][1] == node_id else el1['nodes'][1]
    n2 = el2['nodes'][0] if el2['nodes'][1] == node_id else el2['nodes'][1]

    # 2. Extract raw coordinates
    x0, y0 = nodes[node_id]
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]

    # 3. Calculate raw direction vectors (no unit magnitude scaling!)
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0

    # 4. Exact 2D Cross Product
    cross_product = sp.simplify(dx1 * dy2 - dx2 * dy1)

    return cross_product == 0


def get_node_difficulty_score(node_id, unknown_elements, nodes):
    """
    Evaluates the algebraic difficulty of solving a joint by hand based on 
    geometric coupling.
    
    Returns:
        1: Trivial (1 equation, 1 unknown)
        2: Decoupled (2 unknowns, but at least one is axis-aligned)
        3: Coupled (2 unknowns on diagonals, requires simultaneous equations)
    """
    num_unknowns = len(unknown_elements)
    
    if num_unknowns == 1:
        return 1
        
    if num_unknowns == 2:
        el1, el2 = unknown_elements
        c1, s1 = get_unit_vector(node_id, el1, nodes)
        c2, s2 = get_unit_vector(node_id, el2, nodes)
        
        # If any member is purely horizontal (s=0) or purely vertical (c=0)
        if sp.simplify(c1) == 0 or sp.simplify(s1) == 0 or \
           sp.simplify(c2) == 0 or sp.simplify(s2) == 0:
            return 2
            
        return 3
        
    raise ValueError(f"Cannot score node {node_id}: found {num_unknowns} unknowns.")