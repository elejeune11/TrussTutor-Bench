from collections import deque, defaultdict
import sympy as sp


def solve_global_reactions(nodes, elements, supports, loads):
    """
    Solves for the global reaction forces of a 2D truss.
    Handles both purely numerical inputs and SymPy symbolic inputs.
    
    Parameters:
        nodes (dict): Mapping of node_id -> (X, Y) coordinates.
        elements (list): Included to match the API signature (unused for global statics).
        supports (dict): Boundary conditions (e.g., {1: ['x', 'y'], 2: ['y']}).
        loads (dict): Applied external forces (e.g., {3: (Px, Py)}).
        
    Returns:
        dict: Mapping of string reaction names (e.g., 'R_1_x') to their solved values.
    """
    # 1. Initialize equilibrium equations
    sum_fx = 0
    sum_fy = 0
    sum_mz = 0  # Moment about origin (0,0)
    
    unknown_symbols = []
    reaction_names = []
    
    # 2. Add applied loads to the equations
    for node_id, (fx, fy) in loads.items():
        x, y = nodes[node_id]
        sum_fx += fx
        sum_fy += fy
        # Moment = r x F = (x * Fy) - (y * Fx)
        sum_mz += (x * fy) - (y * fx)
        
    # 3. Add unknown support reactions to the equations
    for node_id, constraints in supports.items():
        x, y = nodes[node_id]
        
        if 'x' in constraints:
            r_x = sp.Symbol(f"R_{node_id}_x")
            unknown_symbols.append(r_x)
            reaction_names.append(f"R_{node_id}_x")
            sum_fx += r_x
            sum_mz += (x * 0) - (y * r_x) # Fy is 0 for an X-reaction
            
        if 'y' in constraints:
            r_y = sp.Symbol(f"R_{node_id}_y")
            unknown_symbols.append(r_y)
            reaction_names.append(f"R_{node_id}_y")
            sum_fy += r_y
            sum_mz += (x * r_y) - (y * 0) # Fx is 0 for a Y-reaction

    # 4. Check for external determinacy
    if len(unknown_symbols) > 3:
        raise ValueError(f"Statically indeterminate externally. Found {len(unknown_symbols)} unknown reactions, maximum is 3.")
    elif len(unknown_symbols) < 3:
        raise ValueError(f"Unstable structure. Found only {len(unknown_symbols)} constraints, minimum is 3.")

    # 5. Solve the system of equations
    # sp.linsolve returns a finite set. We unpack the single tuple inside it.
    solution_set = sp.linsolve([sum_fx, sum_fy, sum_mz], unknown_symbols)
    
    if not solution_set:
        raise ValueError("System is geometrically unstable (equations are dependent).")
        
    solution_tuple = list(solution_set)[0]
    
    # Map the solved expressions back to their string names
    return dict(zip(reaction_names, solution_tuple))


def substitute_values(symbolic_results, substitution_dict):
    """
    Helper function to convert a dictionary of symbolic SymPy results 
    into numerical floats given specific variable values.
    
    Parameters:
        symbolic_results (dict): The output from solve_global_reactions.
        substitution_dict (dict): Mapping of SymPy symbols to numerical values.
        
    Returns:
        dict: The same results evaluated as floats.
    """
    numerical_results = {}
    for key, expression in symbolic_results.items():
        # .subs() injects the numbers, .evalf() ensures it resolves to a float
        numerical_results[key] = float(expression.subs(substitution_dict).evalf())
    return numerical_results


def partition_truss(nodes, elements, cut_element_ids):
    """
    Splits the truss graph by removing the cut elements and finding the 
    resulting disconnected subgraphs using Breadth-First Search (BFS).
    
    Returns:
        tuple: (body_1_nodes, body_2_nodes) as sets of node IDs.
    """
    # Build an adjacency list ignoring the cut elements
    adjacency = defaultdict(list)
    for el in elements:
        if el['id'] not in cut_element_ids:
            n_i, n_j = el['nodes']
            adjacency[n_i].append(n_j)
            adjacency[n_j].append(n_i)
            
    # Find connected components via BFS
    unvisited = set(nodes.keys())
    components = []
    
    while unvisited:
        start_node = unvisited.pop()
        current_component = {start_node}
        queue = deque([start_node])
        
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    current_component.add(neighbor)
                    queue.append(neighbor)
                    
        components.append(current_component)
        
    if len(components) != 2:
        raise ValueError(f"Invalid section cut. The cut must divide the truss into exactly 2 distinct bodies. Found {len(components)} bodies.")
        
    return components[0], components[1]


def solve_method_of_sections(nodes, elements, cut_element_ids, loads, global_reactions, target_fbd_node=None):
    """
    Solves for internal forces of specified cut members using rigid body equilibrium.
    
    Parameters:
        nodes (dict): Node coordinates.
        elements (list): All elements in the truss.
        cut_element_ids (list): IDs of the elements being cut (max 3).
        loads (dict): Applied external loads.
        global_reactions (dict): Previously solved reactions.
        target_fbd_node (any): A node ID specifying which half of the severed truss to evaluate.
        
    Returns:
        dict: Mapping of cut element IDs to their symbolic/numerical forces.
    """
    if len(cut_element_ids) > 3:
        raise ValueError("Method of sections cannot solve for more than 3 unknowns at once.")
        
    # 1. Partition the graph to get our isolated Free Body Diagram
    body_1, body_2 = partition_truss(nodes, elements, cut_element_ids)
    
    # 2. Select the specific Free Body Diagram
    if target_fbd_node is not None:
        if target_fbd_node in body_1:
            fbd_nodes = body_1
        elif target_fbd_node in body_2:
            fbd_nodes = body_2
        else:
            raise ValueError(f"Target FBD node {target_fbd_node} not found in either body.")
    else:
        # Fallback to arbitrary selection if not specified
        fbd_nodes = body_1 
        
    sum_fx = 0
    sum_fy = 0
    sum_mz = 0  # Taking moments about the global origin (0,0)
    
    # 3. Apply Known External Loads on the FBD
    for node_id in fbd_nodes:
        if node_id in loads:
            fx, fy = loads[node_id]
            x, y = nodes[node_id]
            sum_fx += fx
            sum_fy += fy
            sum_mz += (x * fy) - (y * fx)
            
    # 4. Apply Known Global Reactions on the FBD
    # Reactions are passed in as a dict like {'R_1_x': 500, 'R_2_y': 1000}
    for node_id in fbd_nodes:
        x, y = nodes[node_id]
        
        r_x_key = f"R_{node_id}_x"
        if r_x_key in global_reactions:
            r_x = global_reactions[r_x_key]
            sum_fx += r_x
            sum_mz += (x * 0) - (y * r_x)
            
        r_y_key = f"R_{node_id}_y"
        if r_y_key in global_reactions:
            r_y = global_reactions[r_y_key]
            sum_fy += r_y
            sum_mz += (x * r_y) - (y * 0)
            
    # 5. Apply the Unknown Cut Forces
    unknown_symbols = []
    force_names = []
    
    cut_elements = [el for el in elements if el['id'] in cut_element_ids]
    
    for el in cut_elements:
        n_i, n_j = el['nodes']
        
        # Determine which node is inside the FBD and which is outside
        if n_i in fbd_nodes and n_j not in fbd_nodes:
            n_in, n_out = n_i, n_j
        elif n_j in fbd_nodes and n_i not in fbd_nodes:
            n_in, n_out = n_j, n_i
        else:
            raise ValueError(f"Element {el['id']} does not cross the section boundary.")
            
        # Define geometry pointing AWAY from the FBD (assuming Tension is positive)
        x_in, y_in = nodes[n_in]
        x_out, y_out = nodes[n_out]
        
        dx = x_out - x_in
        dy = y_out - y_in
        length = sp.sqrt(dx**2 + dy**2) # sp.sqrt handles both floats and Symbols cleanly
        
        c = dx / length
        s = dy / length
        
        # Define the unknown force variable
        F_cut = sp.Symbol(f"F_{el['id']}")
        unknown_symbols.append(F_cut)
        force_names.append(el['id'])
        
        # Force components
        F_x = F_cut * c
        F_y = F_cut * s
        
        sum_fx += F_x
        sum_fy += F_y
        sum_mz += (x_in * F_y) - (y_in * F_x)
        
    # 6. Solve the 3x3 equilibrium system
    solution_set = sp.linsolve([sum_fx, sum_fy, sum_mz], unknown_symbols)
    
    if not solution_set:
        raise ValueError("Section equilibrium system is singular (e.g., all 3 cut members intersect at a single point).")
        
    solution_tuple = list(solution_set)[0]
    
    return dict(zip(force_names, solution_tuple))


def get_unit_vector(current_node_id, element, nodes):
    """
    Calculates the direction cosines (c, s) for a member, strictly pointing 
    AWAY from the current node towards the opposite node.
    """
    n_i, n_j = element['nodes']
    
    # Identify the node at the other end of the element
    other_node_id = n_j if n_i == current_node_id else n_i
    
    x_curr, y_curr = nodes[current_node_id]
    x_other, y_other = nodes[other_node_id]
    
    dx = x_other - x_curr
    dy = y_other - y_curr
    
    # sp.sqrt safely handles both numeric floats and SymPy symbols
    length = sp.sqrt(dx**2 + dy**2) 
    
    return dx / length, dy / length


def solve_joint_equilibrium(node_id, connected_elements, nodes, known_forces, global_reactions, loads):
    """
    Constructs and solves the 2D concurrent force system (Sum Fx = 0, Sum Fy = 0) 
    for a single joint.
    
    Returns:
        dict: Mapping of newly solved element IDs to their calculated forces.
    """
    sum_fx = 0
    sum_fy = 0
    
    # 1. Apply External Loads
    if node_id in loads:
        fx, fy = loads[node_id]
        sum_fx += fx
        sum_fy += fy
        
    # 2. Apply Global Reactions
    r_x_key = f"R_{node_id}_x"
    if r_x_key in global_reactions:
        sum_fx += global_reactions[r_x_key]
        
    r_y_key = f"R_{node_id}_y"
    if r_y_key in global_reactions:
        sum_fy += global_reactions[r_y_key]
        
    unknown_symbols = []
    force_names = []
    
    # 3. Apply Member Forces (Knowns and Unknowns)
    for el in connected_elements:
        el_id = el['id']
        c, s = get_unit_vector(node_id, el, nodes)
        
        if el_id in known_forces:
            # Multiply the known scalar by the pointing-away unit vector
            F_known = known_forces[el_id]
            sum_fx += F_known * c
            sum_fy += F_known * s
        else:
            # Define a new symbolic unknown (Assumed Tension)
            F_unknown = sp.Symbol(f"F_{el_id}")
            unknown_symbols.append(F_unknown)
            force_names.append(el_id)
            sum_fx += F_unknown * c
            sum_fy += F_unknown * s
            
    # 4. Validation Checks
    if len(unknown_symbols) > 2:
        raise ValueError(f"Cannot solve joint {node_id}: Found {len(unknown_symbols)} unknowns. Max is 2.")
    
    if len(unknown_symbols) == 0:
        return {} # Joint is fully resolved already
        
    # 5. Solve the Local System
    solution_set = sp.linsolve([sum_fx, sum_fy], unknown_symbols)
    
    if not solution_set:
        raise ValueError(f"Equilibrium at joint {node_id} is singular or inconsistent.")
        
    solution_tuple = list(solution_set)[0]
    
    return dict(zip(force_names, solution_tuple))


def are_collinear(node_id, el1, el2, nodes):
    """
    Checks if two members connected to the same node lie on the exact same line of action.
    Uses the cross product of their pointing-away unit vectors.
    """
    c1, s1 = get_unit_vector(node_id, el1, nodes)
    c2, s2 = get_unit_vector(node_id, el2, nodes)
    
    # If the cross product of the vectors is 0, they are collinear (parallel/anti-parallel)
    cross_product = sp.simplify(c1 * s2 - c2 * s1)
    return cross_product == 0


def identify_zfms_by_inspection(nodes, elements, supports, loads, include_collinear_load_rule=False):
    """
    Recursively scans the truss graph to identify Zero-Force Members (ZFMs) 
    using purely geometric rules.
    
    Parameters:
        include_collinear_load_rule (bool): If True, activates Rule 3 (Loaded 2-member joint).
    """
    known_forces = {}
    trace = []
    
    active_element_ids = set(el['id'] for el in elements if el['id'] != 'wall')
    found_new_zfm = True
    
    while found_new_zfm:
        found_new_zfm = False
        
        for node_id in sorted(nodes.keys(), key=str):
            has_support = node_id in supports
            has_load = node_id in loads
            
            # We skip support nodes entirely for pure "by inspection" to be safe
            if has_support:
                continue
                
            connected = [el for el in elements if el['id'] in active_element_ids and node_id in el['nodes']]
            
            # --- UNLOADED JOINT RULES ---
            if not has_load:
                # Rule 1: The Unloaded Corner
                if len(connected) == 2:
                    el1, el2 = connected
                    if not are_collinear(node_id, el1, el2, nodes):
                        known_forces[el1['id']] = 0.0
                        known_forces[el2['id']] = 0.0
                        active_element_ids.remove(el1['id'])
                        active_element_ids.remove(el2['id'])
                        
                        trace.append({
                            'node': node_id,
                            'method': 'By Inspection (Rule 1: Unloaded Corner)',
                            'solved_members': {el1['id']: 0.0, el2['id']: 0.0},
                            'explanation': f"Joint {node_id} has exactly two non-collinear members and no external loads. Both are zero-force members."
                        })
                        found_new_zfm = True
                        break 
                        
                # Rule 2: The T-Joint
                elif len(connected) == 3:
                    el1, el2, el3 = connected
                    zfm_candidate = None
                    
                    if are_collinear(node_id, el1, el2, nodes): zfm_candidate = el3
                    elif are_collinear(node_id, el1, el3, nodes): zfm_candidate = el2
                    elif are_collinear(node_id, el2, el3, nodes): zfm_candidate = el1
                        
                    if zfm_candidate:
                        known_forces[zfm_candidate['id']] = 0.0
                        active_element_ids.remove(zfm_candidate['id'])
                        
                        trace.append({
                            'node': node_id,
                            'method': 'By Inspection (Rule 2: T-Joint)',
                            'solved_members': {zfm_candidate['id']: 0.0},
                            'explanation': f"Joint {node_id} has three members, two of which are collinear. The third non-collinear member ({zfm_candidate['id']}) is a zero-force member."
                        })
                        found_new_zfm = True
                        break 
                        
            # --- LOADED JOINT RULES ---
            elif has_load and include_collinear_load_rule:
                # Rule 3: Collinear Load at a 2-Member Joint
                if len(connected) == 2:
                    el1, el2 = connected
                    fx, fy = loads[node_id]
                    
                    # Ensure it's an actual non-zero load vector
                    if sp.simplify(fx) != 0 or sp.simplify(fy) != 0:
                        c1, s1 = get_unit_vector(node_id, el1, nodes)
                        c2, s2 = get_unit_vector(node_id, el2, nodes)
                        
                        # Mathematical check for collinearity between a member vector and a force vector:
                        # Cross product (Fx*s - Fy*c) == 0
                        zfm_candidate = None
                        if sp.simplify(fx * s1 - fy * c1) == 0:
                            zfm_candidate = el2
                        elif sp.simplify(fx * s2 - fy * c2) == 0:
                            zfm_candidate = el1
                            
                        if zfm_candidate:
                            known_forces[zfm_candidate['id']] = 0.0
                            active_element_ids.remove(zfm_candidate['id'])
                            
                            trace.append({
                                'node': node_id,
                                'method': 'By Inspection (Rule 3: Collinear Load)',
                                'solved_members': {zfm_candidate['id']: 0.0},
                                'explanation': f"Joint {node_id} has two members and an applied load. The load is collinear with one member, so the other non-collinear member ({zfm_candidate['id']}) is a zero-force member."
                            })
                            found_new_zfm = True
                            break

    return known_forces, trace, active_element_ids


def get_node_difficulty_score(node_id, unknown_elements, nodes):
    """
    Evaluates a node to determine the algebraic difficulty of solving its 
    equilibrium equations by hand.
    
    Scores:
        1: Trivial (1 unknown equation).
        2: Decoupled (2 unknowns, but at least one member is purely horizontal or vertical).
        3: Coupled (2 unknowns on diagonals, requiring simultaneous equations).
        
    Returns:
        int: The difficulty score (lower is pedagogically easier).
    """
    num_unknowns = len(unknown_elements)
    
    if num_unknowns == 1:
        return 1
        
    if num_unknowns == 2:
        el1, el2 = unknown_elements
        c1, s1 = get_unit_vector(node_id, el1, nodes)
        c2, s2 = get_unit_vector(node_id, el2, nodes)
        
        # If any member is purely horizontal (s=0) or purely vertical (c=0), 
        # one of the equilibrium equations will isolate the OTHER member.
        # We use sp.simplify to protect against floating-point artifacts.
        if sp.simplify(c1) == 0 or sp.simplify(s1) == 0 or \
           sp.simplify(c2) == 0 or sp.simplify(s2) == 0:
            return 2
            
        # Both members are on diagonals; the equations are coupled.
        return 3
        
    raise ValueError(f"Cannot score node {node_id}: found {num_unknowns} unknowns.")


def solve_method_of_joints(nodes, elements, supports, loads, include_collinear_load_rule=False):
    """
    Pedagogical Method of Joints solver. Generates a step-by-step solution trace 
    and the final internal forces. Matches the input signature of the DSM solver.
    
    Returns:
        tuple: (trace (list), internal_forces (dict), reactions (dict))
    """
    trace = []
    
    # 1. Solve Global Reactions
    # (If the truss is cantilevered and no supports are provided, this returns empty)
    global_reactions = solve_global_reactions(nodes, elements, supports, loads)
    if global_reactions:
        trace.append({
            'step': 'Global Statics',
            'method': 'Rigid Body Equilibrium (Sum Fx, Fy, Mz = 0)',
            'results': global_reactions.copy(),
            'explanation': 'Solved for global support reactions first.'
        })

    # 2. Run the Zero-Force Member (ZFM) Pre-Processor
    known_forces, zfm_trace, active_element_ids = identify_zfms_by_inspection(
        nodes, elements, supports, loads, include_collinear_load_rule
    )
    trace.extend(zfm_trace)
    
    # 3. Main Pathfinding Loop
    while active_element_ids:
        candidate_nodes = []
        
        # Scan all nodes for solvability (<= 2 unknowns)
        for node_id in nodes.keys():
            connected_active = [el for el in elements if el['id'] in active_element_ids and node_id in el['nodes']]
            num_unknowns = len(connected_active)
            
            if 1 <= num_unknowns <= 2:
                score = get_node_difficulty_score(node_id, connected_active, nodes)
                # Tuple structure: (Difficulty Score, String ID for tie-breaking, Node ID, Elements)
                candidate_nodes.append((score, str(node_id), node_id, connected_active))
                
        if not candidate_nodes:
            raise ValueError(
                f"Pathfinding failed. Truss is statically indeterminate internally. "
                f"Remaining unknowns: {active_element_ids}"
            )
            
        # 4. Deterministic Tie-Breaking (Sort by Score Ascending, then Alphanumeric Ascending)
        candidate_nodes.sort(key=lambda x: (x[0], x[1]))
        best_score, _, target_node, target_elements = candidate_nodes[0]
        
        # 5. Solve the selected joint
        # We must pass ALL members connected to this node (both known and unknown)
        all_target_elements = [
            el for el in elements 
            if target_node in el['nodes'] and el['id'] != 'wall'
        ]
        
        newly_solved = solve_joint_equilibrium(
            target_node, all_target_elements, nodes, known_forces, global_reactions, loads
        )
        
        # 6. Process the mathematical results for the Pedagogical Trace
        formatted_results = {}
        for el_id, force_expr in newly_solved.items():
            force_val = sp.simplify(force_expr)
            known_forces[el_id] = force_val
            active_element_ids.remove(el_id)
            
            # Semantic Translation (Tension vs Compression)
            if force_val == 0:
                formatted_results[el_id] = "0.0 (Zero-Force Member)"
            else:
                try:
                    # SymPy can safely check if algebraic expressions with positive=True are negative
                    if force_val.is_negative or (force_val.is_number and float(force_val) < 0):
                        # Strip the negative sign for the human-readable trace
                        formatted_results[el_id] = f"{sp.simplify(-force_val)} (Compression)"
                    else:
                        formatted_results[el_id] = f"{force_val} (Tension)"
                except:
                    # Fallback if evaluation is ambiguous
                    formatted_results[el_id] = f"{force_val} (Algebraic)"
                    
        # Map difficulty score to human-readable explanation
        difficulty_text = {
            1: "Single unknown evaluated directly.",
            2: "Two unknowns with decoupled axes. Evaluated sequentially.",
            3: "Two diagonal unknowns. Evaluated using simultaneous equations."
        }

        trace.append({
            'step': 'Joint Equilibrium',
            'node': target_node,
            'difficulty_tier': best_score,
            'results': formatted_results,
            'explanation': f"Evaluated Joint {target_node}. {difficulty_text[best_score]}"
        })
        
    return trace, known_forces, global_reactions