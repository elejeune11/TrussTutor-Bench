import sympy as sp
from trusstutor_bench.math_helpers import get_unit_vector
from trusstutor_bench.statics_tools import identify_zfms_by_inspection_tool


def corrupted_solve_global_reactions_tool(state):
    """
    Same as solve_global_reactions_tool, but the external moment has the wrong sign,
    simulating a right-hand-rule mistake.

    Args:
        state (TrussState): Truss state, updated in place.
    """
    if not state.supports:
        return

    sum_fx = 0
    sum_fy = 0
    sum_mz = 0

    sorted_supports = sorted(
        state.supports.keys(),
        key=lambda node: len(state.supports[node]),
        reverse=True
    )
    moment_center = sorted_supports[0]
    mx, my = state.nodes[moment_center]
    unknowns = []

    for node_id, (fx, fy) in state.loads.items():
        sum_fx += fx
        sum_fy += fy
        nx, ny = state.nodes[node_id]

        # INTENTIONAL ERROR: the external moment is negated (the correct sign is
        # += (nx - mx) * fy - (ny - my) * fx).
        sum_mz -= ((nx - mx) * fy - (ny - my) * fx)

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

    if len(unknowns) > 3:
        raise ValueError(f"Truss is globally statically indeterminate: {len(unknowns)} unknowns.")

    active_equations = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        clean_eq = sp.nsimplify(eq, tolerance=1e-9, rational=True)
        if clean_eq != 0:
            active_equations.append(clean_eq)
    solution = sp.linsolve(active_equations, unknowns)

    if not solution:
        raise ValueError("Global equilibrium system is singular or inconsistent.")

    solution_vals = list(solution)[0]

    results = {}
    for var, val in zip(unknowns, solution_vals):
        simplified_val = sp.simplify(val)
        state.global_reactions[var.name] = simplified_val
        results[var.name] = simplified_val

    # The explanation stays confident, like a student who did not notice the error.
    state.log_step(
        step_type="Global Statics",
        target="Entire Truss",
        results=results,
        explanation=f"Solved rigid-body equilibrium using Sum Fx, Sum Fy, and Sum Mz about Node {moment_center}."
    )


def corrupted_solve_single_joint_tool(state, node_id):
    """
    Same as solve_single_joint_tool, but an overdetermined joint is solved from one equation,
    ignoring the contradiction, like a student who does not check the last equations.

    Args:
        state (TrussState): Truss state, updated in place.
        node_id: Joint to solve.

    Returns:
        bool: True if the joint was solved, False otherwise.
    """
    if node_id in state.supports:
        for dof in state.supports[node_id]:
            reaction_key = f"R_{node_id}_{dof}"
            if reaction_key not in state.global_reactions:
                return False

    connected_elements = [el for el in state.elements if node_id in el['nodes']]
    unknown_elements = [el for el in connected_elements if el['id'] in state.active_element_ids]
    known_elements = [el for el in connected_elements if el['id'] not in state.active_element_ids]

    num_unknowns = len(unknown_elements)
    if num_unknowns == 0 or num_unknowns > 2:
        return False

    sum_fx, sum_fy = 0, 0

    if node_id in state.loads:
        fx, fy = state.loads[node_id]
        sum_fx += fx
        sum_fy += fy

    if f"R_{node_id}_x" in state.global_reactions:
        sum_fx += state.global_reactions[f"R_{node_id}_x"]
    if f"R_{node_id}_y" in state.global_reactions:
        sum_fy += state.global_reactions[f"R_{node_id}_y"]

    for el in known_elements:
        c, s = get_unit_vector(node_id, el, state.nodes)
        force_val = state.known_forces[el['id']]
        sum_fx += force_val * c
        sum_fy += force_val * s

    unknown_vars = []
    for el in unknown_elements:
        c, s = get_unit_vector(node_id, el, state.nodes)
        var_name = f"F_{el['id']}"
        F_var = sp.Symbol(var_name)
        unknown_vars.append(F_var)
        sum_fx += F_var * c
        sum_fy += F_var * s

    active_equations = []
    for eq in [sum_fx, sum_fy]:
        clean_eq = sp.simplify(eq)
        if clean_eq != 0:
            active_equations.append(clean_eq)

    # INTENTIONAL ERROR: with more equations than unknowns, solve one equation and ignore the contradiction.
    solution = None
    if len(active_equations) > num_unknowns:
        for eq in active_equations:
            sol = sp.linsolve([eq], unknown_vars)
            if sol:
                solution = sol
                break
    else:
        if len(active_equations) >= num_unknowns:
            solution = sp.linsolve(active_equations, unknown_vars)

    if not solution:
        return False

    solution_vals = list(solution)[0]
    results = {}
    for el, val in zip(unknown_elements, solution_vals):
        simplified_val = sp.simplify(val)
        state.known_forces[el['id']] = simplified_val
        state.active_element_ids.remove(el['id'])
        results[el['id']] = simplified_val

    state.log_step(
        step_type="Method of Joints",
        target=node_id,
        results=results,
        explanation=f"Evaluated joint {node_id}. Solved using Sum Fx = 0 and Sum Fy = 0."
    )
    return True


def corrupted_autonomous_moj_macro(state, include_collinear_load_rule=False):
    """
    Same as autonomous_moj_macro, but using the corrupted reactions and joint tools.

    Args:
        state (TrussState): Truss state, updated in place.
        include_collinear_load_rule (bool): Whether to apply rule 3 (collinear load at a joint).

    Returns:
        bool: True if every member was solved.
    """
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (
            sp.nsimplify(fx, rational=True),
            sp.nsimplify(fy, rational=True)
        )

    try:
        corrupted_solve_global_reactions_tool(state)
    except ValueError as e:
        state.log_step(step_type="Error", target="Global Reactions", results={}, explanation=str(e))
        return False

    identify_zfms_by_inspection_tool(state, include_collinear_load_rule)

    made_progress = True
    while made_progress and state.active_element_ids:
        made_progress = False
        for node_id in sorted(state.nodes.keys(), key=str):
            # INTENTIONAL ERROR: uses the corrupted joint tool.
            if corrupted_solve_single_joint_tool(state, node_id):
                made_progress = True
                identify_zfms_by_inspection_tool(state, include_collinear_load_rule)
                break

    return len(state.active_element_ids) == 0


def corrupted_solve_method_of_sections_tool(state, cut_element_ids):
    """
    Same as solve_method_of_sections_tool, but the signs of the answers are flipped,
    simulating a tension/compression mix-up.

    Args:
        state (TrussState): Truss state, updated in place.
        cut_element_ids (list): IDs of the members the cut passes through.

    Returns:
        bool: True if solved, False if the cut is invalid or unsolvable.
    """
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (sp.nsimplify(fx, rational=True), sp.nsimplify(fy, rational=True))

    state.nodes = {
        nid: (sp.nsimplify(x, rational=True), sp.nsimplify(y, rational=True))
        for nid, (x, y) in state.nodes.items()
    }
    unknown_cut_ids = [eid for eid in cut_element_ids if eid in state.active_element_ids]
    if len(unknown_cut_ids) == 0 or len(unknown_cut_ids) > 3:
        return False

    adjacency = {node: [] for node in state.nodes}
    for el in state.elements:
        if el['id'] not in cut_element_ids:
            n1, n2 = el['nodes']
            adjacency[n1].append(n2)
            adjacency[n2].append(n1)

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

    if len(components) != 2:
        return False

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
        return False

    sum_fx = 0
    sum_fy = 0
    sum_mz = 0

    moment_center = list(sub_truss_nodes)[0]
    mx, my = state.nodes[moment_center]

    for node in sub_truss_nodes:
        if node in state.loads:
            fx, fy = state.loads[node]
            sum_fx += fx
            sum_fy += fy
            nx, ny = state.nodes[node]
            sum_mz += (nx - mx) * fy - (ny - my) * fx

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

    unknown_vars = []
    for eid in cut_element_ids:
        el = next(e for e in state.elements if e['id'] == eid)
        n1, n2 = el['nodes']

        if n1 in sub_truss_nodes and n2 not in sub_truss_nodes:
            local_node, target_node = n1, n2
        elif n2 in sub_truss_nodes and n1 not in sub_truss_nodes:
            local_node, target_node = n2, n1
        else:
            return False

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

    active_eqs = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        clean_eq = sp.nsimplify(eq, tolerance=1e-9, rational=True)
        if clean_eq != 0:
            active_eqs.append(clean_eq)

    solution = sp.linsolve(active_eqs, unknown_vars)

    if not solution:
        return False

    solution_vals = list(solution)[0]

    results = {}
    for eid, val in zip(unknown_cut_ids, solution_vals):
        # INTENTIONAL ERROR: the sign of the answer is flipped.
        simplified_val = -sp.simplify(val)

        state.known_forces[eid] = simplified_val
        state.active_element_ids.remove(eid)
        results[eid] = simplified_val

    state.mos_cut_elements = cut_element_ids
    state.mos_target_node = list(sub_truss_nodes)[0]

    state.log_step(
        step_type="Method of Sections",
        target=f"Cut: {', '.join(map(str, cut_element_ids))}",
        results=results,
        explanation=f"Evaluated FBD of sub-truss nodes {sorted(list(sub_truss_nodes))} using Sum Fx, Sum Fy, and Sum Mz."
    )

    return True
