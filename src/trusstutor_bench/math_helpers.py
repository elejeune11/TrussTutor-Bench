import sympy as sp


def get_unit_vector(node_id, element, nodes):
    """
    Returns the direction cosines (c, s) of a member, pointing away from the
    given node (the tension convention).

    Args:
        node_id: Node the vector starts from.
        element (dict): Element with "nodes".
        nodes (dict): Node ID -> (x, y).

    Returns:
        tuple: (c, s).
    """
    n1, n2 = element['nodes']
    target_node = n2 if n1 == node_id else n1

    x1, y1 = nodes[node_id]
    x2, y2 = nodes[target_node]

    dx = sp.simplify(x2 - x1)
    dy = sp.simplify(y2 - y1)

    # Simplify the squared length before the square root so trig identities reduce it cleanly.
    length_squared = sp.simplify(dx**2 + dy**2)
    length = sp.sqrt(length_squared)

    c = sp.simplify(dx / length)
    s = sp.simplify(dy / length)

    return c, s


def are_collinear(node_id, el1, el2, nodes):
    """
    Checks whether two elements are collinear at a node, using the exact 2D
    cross product of the coordinate differences.

    Args:
        node_id: Node shared by both elements.
        el1 (dict): First element.
        el2 (dict): Second element.
        nodes (dict): Node ID -> (x, y).

    Returns:
        bool: True if the elements are collinear.
    """
    n1 = el1['nodes'][0] if el1['nodes'][1] == node_id else el1['nodes'][1]
    n2 = el2['nodes'][0] if el2['nodes'][1] == node_id else el2['nodes'][1]

    x0, y0 = nodes[node_id]
    x1, y1 = nodes[n1]
    x2, y2 = nodes[n2]

    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0

    cross_product = sp.simplify(dx1 * dy2 - dx2 * dy1)

    return cross_product == 0


def get_node_difficulty_score(node_id, unknown_elements, nodes):
    """
    Scores how hard it is to solve a joint by hand, from how its unknown members are oriented.

    Args:
        node_id: Node being solved.
        unknown_elements (list): Elements at the node with unknown force (1 or 2).
        nodes (dict): Node ID -> (x, y).

    Returns:
        int: 1 for one unknown, 2 for two unknowns with at least one axis-aligned,
            3 for two diagonal unknowns that need simultaneous equations.
    """
    num_unknowns = len(unknown_elements)

    if num_unknowns == 1:
        return 1

    if num_unknowns == 2:
        el1, el2 = unknown_elements
        c1, s1 = get_unit_vector(node_id, el1, nodes)
        c2, s2 = get_unit_vector(node_id, el2, nodes)

        if sp.simplify(c1) == 0 or sp.simplify(s1) == 0 or \
           sp.simplify(c2) == 0 or sp.simplify(s2) == 0:
            return 2

        return 3

    raise ValueError(f"Cannot score node {node_id}: found {num_unknowns} unknowns.")
