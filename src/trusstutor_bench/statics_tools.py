import sympy as sp
from .math_helpers import are_collinear, get_unit_vector, get_node_difficulty_score


def solve_global_reactions_tool(state):
    """
    Calculates rigid-body support reactions and mutates the TrussState directly.
    Logs the step to the pedagogical trace.
    """
    if not state.supports:
        # If there are no supports (e.g., purely floating or a sub-section), skip.
        return 

    sum_fx = 0
    sum_fy = 0
    sum_mz = 0

    # Choose a moment center pedagogically.
    # We prioritize the support with the most unknown reactions (e.g., a pin with 2 
    # unknowns over a roller with 1) to eliminate them from the moment equation.
    sorted_supports = sorted(
        state.supports.keys(), 
        key=lambda node: len(state.supports[node]), 
        reverse=True
    )
    moment_center = sorted_supports[0]
    
    mx, my = state.nodes[moment_center]

    unknowns = []

    # 1. Apply external loads
    for node_id, (fx, fy) in state.loads.items():
        sum_fx += fx
        sum_fy += fy
        nx, ny = state.nodes[node_id]
        
        # Cross product for 2D moment: (rx * Fy) - (ry * Fx)
        sum_mz += (nx - mx) * fy - (ny - my) * fx

    # 2. Apply unknown support reactions
    for node_id, directions in state.supports.items():
        nx, ny = state.nodes[node_id]
        for direction in directions:
            var_name = f"R_{node_id}_{direction}"
            R_var = sp.Symbol(var_name)
            unknowns.append(R_var)

            if direction == 'x':
                sum_fx += R_var
                sum_mz += -(ny - my) * R_var
            elif direction == 'y':
                sum_fy += R_var
                sum_mz += (nx - mx) * R_var

    # 3. Solve the rigid-body system
    # (Global statics strictly limits us to 3 unknowns in 2D)
    if len(unknowns) > 3:
        raise ValueError(f"Truss is globally statically indeterminate: {len(unknowns)} unknowns.")
    
    # Filter out trivial 0=0 equations to prevent SymPy linsolve errors
    active_equations = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        # Convert float noise like 1e-15 to 0, and map decimals to exact rationals
        clean_eq = sp.nsimplify(eq, tolerance=1e-9, rational=True)
        if clean_eq != 0:
            active_equations.append(clean_eq)
    solution = sp.linsolve(active_equations, unknowns)

    if not solution:
        raise ValueError("Global equilibrium system is singular or inconsistent.")

    solution_vals = list(solution)[0]

    # 4. Mutate the State and Log the Trace
    results = {}
    for var, val in zip(unknowns, solution_vals):
        simplified_val = sp.simplify(val)
        state.global_reactions[var.name] = simplified_val
        results[var.name] = simplified_val

    state.log_step(
        step_type="Global Statics",
        target="Entire Truss",
        results=results,
        explanation=f"Solved rigid-body equilibrium using Sum Fx, Sum Fy, and Sum Mz about Node {moment_center}."
    )


def identify_zfms_by_inspection_tool(state, include_collinear_load_rule=False):
    """
    Recursively scans the state to identify Zero-Force Members (ZFMs) using purely 
    geometric rules. Mutates the TrussState directly.
    Returns True if NEW zero-force members were identified, False otherwise.
    """
    def is_zero(val):
        """Safely checks if a SymPy expression or float is effectively zero."""
        try:
            return abs(float(val)) < 1e-9
        except TypeError:
            # Fallback for complex unresolved algebraic sympy symbols
            return sp.simplify(val) == 0

    found_new_zfm = True
    progress_made_overall = False
    
    # Loop continues as long as finding a ZFM exposes a new one at an adjacent node
    while found_new_zfm:
        found_new_zfm = False
        
        for node_id in sorted(state.nodes.keys(), key=str):
            if node_id in state.supports:
                continue
                
            # 1. Gather ALL physically connected elements
            all_connected = [el for el in state.elements if node_id in el['nodes']]
            
            # 2. Filter into knowns and unknowns
            unknown_elements = [el for el in all_connected if el['id'] in state.active_element_ids]
            known_elements = [el for el in all_connected if el['id'] not in state.active_element_ids]
            
            if not unknown_elements:
                continue 
                
            # 3. Evaluate External Loads (Using strict zero check)
            fx, fy = 0, 0
            has_ext_load = False
            if node_id in state.loads:
                fx, fy = state.loads[node_id]
                if not is_zero(fx) or not is_zero(fy):
                    has_ext_load = True
                    
            # 4. Evaluate Internal Loads (Using strict zero check)
            has_int_load = False
            for el in known_elements:
                if not is_zero(state.known_forces[el['id']]):
                    has_int_load = True
                    break
                    
            is_unloaded = not has_ext_load and not has_int_load
            
            # --- UNLOADED JOINT RULES ---
            if is_unloaded:
                # Rule 1: The Unloaded Corner
                if len(unknown_elements) == 2:
                    el1, el2 = unknown_elements
                    if not are_collinear(node_id, el1, el2, state.nodes):
                        state.known_forces[el1['id']] = 0.0
                        state.known_forces[el2['id']] = 0.0
                        state.active_element_ids.remove(el1['id'])
                        state.active_element_ids.remove(el2['id'])
                        
                        state.log_step(
                            step_type="By Inspection (Rule 1: Unloaded Corner)",
                            target=node_id,
                            results={el1['id']: 0.0, el2['id']: 0.0},
                            explanation=f"Joint {node_id} has exactly two non-collinear active members and no loads. Both are zero-force members."
                        )
                        found_new_zfm = True
                        progress_made_overall = True
                        break 
                        
                # Rule 2: The T-Joint
                elif len(unknown_elements) == 3:
                    el1, el2, el3 = unknown_elements
                    zfm_candidate = None
                    
                    if are_collinear(node_id, el1, el2, state.nodes): zfm_candidate = el3
                    elif are_collinear(node_id, el1, el3, state.nodes): zfm_candidate = el2
                    elif are_collinear(node_id, el2, el3, state.nodes): zfm_candidate = el1
                        
                    if zfm_candidate:
                        state.known_forces[zfm_candidate['id']] = 0.0
                        state.active_element_ids.remove(zfm_candidate['id'])
                        
                        state.log_step(
                            step_type="By Inspection (Rule 2: T-Joint)",
                            target=node_id,
                            results={zfm_candidate['id']: 0.0},
                            explanation=f"Joint {node_id} has three active members, two of which are collinear. The third non-collinear member ({zfm_candidate['id']}) is a zero-force member."
                        )
                        found_new_zfm = True
                        progress_made_overall = True
                        break 
                        
            # --- LOADED JOINT RULES ---
            elif has_ext_load and not has_int_load and include_collinear_load_rule:
                # Rule 3: Collinear Load at a 2-Member Joint
                if len(unknown_elements) == 2:
                    el1, el2 = unknown_elements
                    
                    c1, s1 = get_unit_vector(node_id, el1, state.nodes)
                    c2, s2 = get_unit_vector(node_id, el2, state.nodes)
                    
                    zfm_candidate = None
                    if is_zero(fx * s1 - fy * c1):
                        zfm_candidate = el2
                    elif is_zero(fx * s2 - fy * c2):
                        zfm_candidate = el1
                        
                    if zfm_candidate:
                        state.known_forces[zfm_candidate['id']] = 0.0
                        state.active_element_ids.remove(zfm_candidate['id'])
                        
                        state.log_step(
                            step_type="By Inspection (Rule 3: Collinear Load)",
                            target=node_id,
                            results={zfm_candidate['id']: 0.0},
                            explanation=f"Joint {node_id} has two active members and an applied load collinear with one. The other member ({zfm_candidate['id']}) is a zero-force member."
                        )
                        found_new_zfm = True
                        progress_made_overall = True
                        break

    return progress_made_overall


def solve_single_joint_tool(state, node_id):
    """
    Evaluates a single node using 2D particle equilibrium. 
    Returns True if the joint was successfully solved and state was mutated, 
    Returns False if the joint has > 2 unknowns, is underdetermined, or is already solved.
    """
    # 1. NEW SAFETY CHECK: Prevent "Invisible Supports"
    # If this node is a support, and we don't have the reactions yet, we cannot solve it here.
    if node_id in state.supports:
        for dof in state.supports[node_id]:
            reaction_key = f"R_{node_id}_{dof}"
            if reaction_key not in state.global_reactions:
                return False 

    # 2. Gather all connected elements
    connected_elements = [el for el in state.elements if node_id in el['nodes']]
    
    # Separate into knowns and unknowns
    unknown_elements = [el for el in connected_elements if el['id'] in state.active_element_ids]
    known_elements = [el for el in connected_elements if el['id'] not in state.active_element_ids]
    
    num_unknowns = len(unknown_elements)
    
    if num_unknowns == 0 or num_unknowns > 2:
        return False
        
    difficulty = get_node_difficulty_score(node_id, unknown_elements, state.nodes)

    sum_fx = 0
    sum_fy = 0

    # 3. Apply external loads
    if node_id in state.loads:
        fx, fy = state.loads[node_id]
        sum_fx += fx
        sum_fy += fy

    # 4. Apply global support reactions 
    if f"R_{node_id}_x" in state.global_reactions:
        sum_fx += state.global_reactions[f"R_{node_id}_x"]
    if f"R_{node_id}_y" in state.global_reactions:
        sum_fy += state.global_reactions[f"R_{node_id}_y"]

    # 5. Apply internal forces from previously solved members
    for el in known_elements:
        c, s = get_unit_vector(node_id, el, state.nodes)
        force_val = state.known_forces[el['id']]
        sum_fx += force_val * c
        sum_fy += force_val * s

    # 6. Set up equations for the unknown members
    unknown_vars = []
    for el in unknown_elements:
        c, s = get_unit_vector(node_id, el, state.nodes)
        var_name = f"F_{el['id']}"
        F_var = sp.Symbol(var_name)
        unknown_vars.append(F_var)
        
        sum_fx += F_var * c
        sum_fy += F_var * s

    # 7. Solve the system
    active_equations = []
    for eq in [sum_fx, sum_fy]:
        # Use sp.simplify to algebraically group terms without forcing 
        # rational approximations of exact irrationals like sqrt(3)
        clean_eq = sp.simplify(eq)
        if clean_eq != 0:
            active_equations.append(clean_eq)

    # NEW SAFETY CHECK: Prevent parameterized solutions from polluting the state
    if len(active_equations) < num_unknowns:
        return False

    solution = sp.linsolve(active_equations, unknown_vars)

    if not solution:
        return False

    solution_vals = list(solution)[0]

    # 8. Mutate State and Log Trace
    results = {}
    for el, val in zip(unknown_elements, solution_vals):
        simplified_val = sp.simplify(val)
        state.known_forces[el['id']] = simplified_val
        state.active_element_ids.remove(el['id'])
        results[el['id']] = simplified_val

    difficulty_label = {1: "Trivial", 2: "Decoupled", 3: "Coupled"}[difficulty]

    state.log_step(
        step_type="Method of Joints",
        target=node_id,
        results=results,
        explanation=f"Evaluated joint {node_id} ({difficulty_label} System: {num_unknowns} unknowns). Solved using Sum Fx = 0 and Sum Fy = 0."
    )
    
    return True


def solve_method_of_sections_tool(state, cut_element_ids):
    """
    Evaluates a free body diagram of a sub-truss created by severing specific members.
    Returns True if successfully solved, False if the cut is invalid or unsolvable.
    """
    # Clean loads for moment arms calculations
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (sp.nsimplify(fx, rational=True), sp.nsimplify(fy, rational=True))
        
    # Also ensure node coordinates are exact for lever arm calculations
    state.nodes = {
        nid: (sp.nsimplify(x, rational=True), sp.nsimplify(y, rational=True)) 
        for nid, (x, y) in state.nodes.items()
    }
    # 1. Verify we are not trying to solve too many unknowns
    unknown_cut_ids = [eid for eid in cut_element_ids if eid in state.active_element_ids]
    if len(unknown_cut_ids) == 0 or len(unknown_cut_ids) > 3:
        return False
        
    # 2. Build a graph to find the two separated sub-trusses
    # We use all elements EXCEPT the ones we are actively cutting
    adjacency = {node: [] for node in state.nodes}
    for el in state.elements:
        if el['id'] not in cut_element_ids:
            n1, n2 = el['nodes']
            adjacency[n1].append(n2)
            adjacency[n2].append(n1)
            
    # Find connected components using Breadth-First Search
    visited = set()
    components = []
    for node in state.nodes:
        if node not in visited:
            comp = set()
            queue = [node]
            while queue:
                curr = queue.pop(0)
                if curr not in comp:
                    comp.add(curr)
                    visited.add(curr)
                    queue.extend([n for n in adjacency[curr] if n not in visited])
            components.append(comp)
            
    # A textbook Method of Sections cut must divide the truss into exactly two bodies
    if len(components) != 2:
        return False
        
    # 3. Choose the solvable sub-truss
    # A sub-truss is only solvable if ALL of its support reactions are already known.
    def is_sub_truss_solvable(nodes):
        for node in nodes:
            if node in state.supports:
                for direction in state.supports[node]:
                    if f"R_{node}_{direction}" not in state.global_reactions:
                        return False
        return True

    if is_sub_truss_solvable(components[0]):
        sub_truss_nodes = components[0]
    elif is_sub_truss_solvable(components[1]):
        sub_truss_nodes = components[1]
    else:
        # Neither side can be solved yet (e.g. both sides have unknown reactions)
        return False

    sum_fx = 0
    sum_fy = 0
    sum_mz = 0
    
    # Pick an arbitrary node in the sub-truss for the moment center
    moment_center = list(sub_truss_nodes)[0] 
    mx, my = state.nodes[moment_center]

    # 4. Apply external loads acting on the chosen sub-truss
    for node in sub_truss_nodes:
        if node in state.loads:
            fx, fy = state.loads[node]
            sum_fx += fx
            sum_fy += fy
            nx, ny = state.nodes[node]
            sum_mz += (nx - mx) * fy - (ny - my) * fx

    # 5. Apply known global reactions acting on the chosen sub-truss
    for node in sub_truss_nodes:
        if f"R_{node}_x" in state.global_reactions:
            rx = state.global_reactions[f"R_{node}_x"]
            sum_fx += rx
            nx, ny = state.nodes[node]
            sum_mz += -(ny - my) * rx
        if f"R_{node}_y" in state.global_reactions:
            ry = state.global_reactions[f"R_{node}_y"]
            sum_fy += ry
            nx, ny = state.nodes[node]
            sum_mz += (nx - mx) * ry

    # 6. Apply the severed cut forces (the unknowns and any already knowns)
    unknown_vars = []
    for eid in cut_element_ids:
        el = next(e for e in state.elements if e['id'] == eid)
        n1, n2 = el['nodes']
        
        # Determine which node of the member is inside the sub-truss
        if n1 in sub_truss_nodes and n2 not in sub_truss_nodes:
            local_node, target_node = n1, n2
        elif n2 in sub_truss_nodes and n1 not in sub_truss_nodes:
            local_node, target_node = n2, n1
        else:
            return False # Member doesn't cross the boundary perfectly
            
        c, s = get_unit_vector(local_node, el, state.nodes)
        nx, ny = state.nodes[local_node]
        
        if eid in state.known_forces:
            force_val = state.known_forces[eid]
        else:
            var_name = f"F_{eid}"
            force_val = sp.Symbol(var_name)
            unknown_vars.append(force_val)
            
        sum_fx += force_val * c
        sum_fy += force_val * s
        sum_mz += (nx - mx) * (force_val * s) - (ny - my) * (force_val * c)

    # 7. Solve the 3-equation system
    active_eqs = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        # Snap float noise to zero and convert decimals to exact rational fractions
        clean_eq = sp.nsimplify(eq, tolerance=1e-9, rational=True)
        if clean_eq != 0:
            active_eqs.append(clean_eq)

    solution = sp.linsolve(active_eqs, unknown_vars)

    if not solution:
        return False

    if not solution:
        return False

    solution_vals = list(solution)[0]

    # 8. Mutate State and Log
    results = {}
    for eid, val in zip(unknown_cut_ids, solution_vals):
        simplified_val = sp.simplify(val)
        state.known_forces[eid] = simplified_val
        state.active_element_ids.remove(eid)
        results[eid] = simplified_val

    # --- NEW: Save the cut data for the FBD generator ---
    state.mos_cut_elements = cut_element_ids
    
    # We just need one node that exists inside the kept sub-truss to anchor the FBD camera
    state.mos_target_node = list(sub_truss_nodes)[0]
    # ----------------------------------------------------

    state.log_step(
        step_type="Method of Sections",
        target=f"Cut: {', '.join(map(str, cut_element_ids))}",
        results=results,
        explanation=f"Evaluated FBD of sub-truss nodes {sorted(list(sub_truss_nodes))} using Sum Fx, Sum Fy, and Sum Mz."
    )
    
    return True

def autonomous_moj_macro(state, include_collinear_load_rule=False):
    """
    Orchestrates the complete Method of Joints process.
    Sequentially applies global reactions, ZFM inspection, and single-joint 
    solvers until the truss is fully solved or no further progress can be made.
    
    Returns True if the truss is completely solved, False if internally indeterminate.
    """
    # Clean inputs before starting math
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (
            sp.nsimplify(fx, rational=True), 
            sp.nsimplify(fy, rational=True)
        )

    # 1. Attempt global reactions first
    try:
        solve_global_reactions_tool(state)
    except ValueError as e:
        state.log_step(
            step_type="Error", 
            target="Global Reactions", 
            results={}, 
            explanation=str(e)
        )
        return False

    # 2. Initial ZFM sweep
    identify_zfms_by_inspection_tool(state, include_collinear_load_rule)

    # 3. Main Orchestration Loop
    made_progress = True
    
    # Continue as long as we made progress in the last pass AND there are unsolved members
    while made_progress and state.active_element_ids:
        made_progress = False

        # Sort nodes for deterministic solving order (crucial for reproducible pedagogical traces)
        for node_id in sorted(state.nodes.keys(), key=str):
            
            # Attempt to solve the joint
            if solve_single_joint_tool(state, node_id):
                made_progress = True
                
                # Pedagogical optimization: After a successful joint solve, a new ZFM 
                # might be exposed. Run a quick sweep to catch cascading members.
                identify_zfms_by_inspection_tool(state, include_collinear_load_rule)
                
                # Break the inner for-loop to restart the node scan from the beginning.
                # This prioritizes newly unlocked, simpler joints.
                break

    # Return True only if every single element was successfully solved
    return len(state.active_element_ids) == 0


def programmatic_solution(solution_recipe, sampled_vars):
    """
    Orchestrates the execution of a declarative solution recipe to solve a benchmark problem.
    
    This function acts as a strict, chronological pipeline runner, completely abstracting 
    the execution loop away from individual problem templates. It guarantees execution 
    order by sorting the recipe keys and enforces a three-phase lifecycle:
    
    1. The Creator (First Step): Dynamically injects `sampled_vars` alongside static 
       geometry/load inputs to initialize and return the core `TrussState` object.
    2. The Mutators (Middle Steps): Sequentially passes the `TrussState` object through 
       a series of physics tools, modifying the state in place.
    3. The Extractor (Final Step): Evaluates the fully resolved `TrussState` to pull 
       the specific target metrics defined by the problem schema.
    
    Args:
        solution_recipe (dict): A numbered execution ledger defining the pedagogical path. 
            Must follow the structure:
            {
                1: {"function": callable, "inputs": {static_kwargs}},
                2: {"function": callable, "inputs": {static_kwargs}},
                ...
            }
        sampled_vars (dict): The dynamic numerical values assigned to the problem's 
            symbolic variables for this specific execution instance (e.g., {"P": 50.0}).
            
    Returns:
        dict: The final extracted ground-truth solution, structured according to the 
            template's specific expected output schema.
            
    Raises:
        RuntimeError: If any intermediate solver step returns `False`, indicating 
            that the physics engine failed to resolve the requested state.
    """
    # Safely sort the dictionary keys to guarantee chronological execution
    sorted_keys = sorted(solution_recipe.keys())
    steps = [solution_recipe[k] for k in sorted_keys]
    
    # 1. The Creator
    init_step = steps[0]
    init_func = init_step["function"]
    init_inputs = init_step["inputs"]
    
    state = init_func(sampled_vars=sampled_vars, **init_inputs)
    
    # 2. The Mutators
    solver_steps = steps[1:-1]
    
    for current_step in solver_steps:
        solver_func = current_step["function"]
        solver_inputs = current_step["inputs"]
        
        success = solver_func(state, **solver_inputs)
        
        if success is False:
            raise RuntimeError(f"Pipeline failed at step: {solver_func.__name__}")
            
    # 3. The Extractor
    ext_step = steps[-1]
    ext_func = ext_step["function"]
    ext_inputs = ext_step["inputs"]
    
    ground_truth = ext_func(state, **ext_inputs)
    
    return ground_truth