import math
import sympy as sp
from .math_helpers import are_collinear, get_unit_vector, get_node_difficulty_score


def solve_global_reactions_tool(state):
    """
    Solves the support reactions from rigid-body equilibrium and stores them in the state.
    Does nothing if there are no supports.

    Args:
        state (TrussState): Truss state, updated in place.
    """
    if not state.supports:
        return

    sum_fx = 0
    sum_fy = 0
    sum_mz = 0

    # Moment center: the support with the most unknown reactions, which removes them from the moment equation.
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

        # 2D moment: (rx * Fy) - (ry * Fx)
        sum_mz += (nx - mx) * fy - (ny - my) * fx

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

    # Global statics in 2D allows at most 3 unknowns.
    if len(unknowns) > 3:
        raise ValueError(f"Truss is globally statically indeterminate: {len(unknowns)} unknowns.")

    # Drop 0 = 0 equations, which would break linsolve.
    active_equations = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        # nsimplify turns float noise into 0 and decimals into exact rationals.
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

    state.log_step(
        step_type="Global Statics",
        target="Entire Truss",
        results=results,
        explanation=f"Solved rigid-body equilibrium using Sum Fx, Sum Fy, and Sum Mz about Node {moment_center}."
    )


def identify_zfms_by_inspection_tool(state, include_collinear_load_rule=False):
    """
    Marks zero-force members (ZFMs) using three geometric rules, repeating until no new ones appear.

    Args:
        state (TrussState): Truss state, updated in place.
        include_collinear_load_rule (bool): Whether to apply rule 3 (collinear load at a joint).

    Returns:
        bool: True if new zero-force members were found.
    """
    def is_zero(val):
        """
        Checks whether a SymPy expression or float is effectively zero.

        Args:
            val: Expression or number.

        Returns:
            bool: True if zero.
        """
        try:
            return abs(float(val)) < 1e-9
        except TypeError:
            return sp.simplify(val) == 0

    found_new_zfm = True
    progress_made_overall = False

    # Repeat while a zero-force member may expose another at a neighbouring node.
    while found_new_zfm:
        found_new_zfm = False

        for node_id in sorted(state.nodes.keys(), key=str):
            if node_id in state.supports:
                continue

            all_connected = [el for el in state.elements if node_id in el['nodes']]

            unknown_elements = [el for el in all_connected if el['id'] in state.active_element_ids]
            known_elements = [el for el in all_connected if el['id'] not in state.active_element_ids]

            if not unknown_elements:
                continue

            fx, fy = 0, 0
            has_ext_load = False
            if node_id in state.loads:
                fx, fy = state.loads[node_id]
                if not is_zero(fx) or not is_zero(fy):
                    has_ext_load = True

            has_int_load = False
            for el in known_elements:
                if not is_zero(state.known_forces[el['id']]):
                    has_int_load = True
                    break

            is_unloaded = not has_ext_load and not has_int_load

            if is_unloaded:
                # Rule 1: unloaded corner
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

                # Rule 2: T-joint
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

            elif has_ext_load and not has_int_load and include_collinear_load_rule:
                # Rule 3: collinear load at a two-member joint
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
    Solves one joint with at most two unknown members using Sum Fx = 0 and Sum Fy = 0.

    Args:
        state (TrussState): Truss state, updated in place.
        node_id: Joint to solve.

    Returns:
        bool: True if the joint was solved, False if it has more than two unknowns,
            is underdetermined, or is already solved.
    """
    # A support whose reactions are unknown cannot be solved yet.
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

    difficulty = get_node_difficulty_score(node_id, unknown_elements, state.nodes)

    sum_fx = 0
    sum_fy = 0

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
        # simplify keeps exact irrationals such as sqrt(3), unlike nsimplify.
        clean_eq = sp.simplify(eq)
        if clean_eq != 0:
            active_equations.append(clean_eq)

    # Fewer equations than unknowns gives a parametric solution, so skip it.
    if len(active_equations) < num_unknowns:
        return False

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
    Solves the cut members from a free body diagram of one side of the cut.

    Args:
        state (TrussState): Truss state, updated in place.
        cut_element_ids (list): IDs of the members the cut passes through.

    Returns:
        bool: True if solved, False if the cut is invalid or unsolvable.
    """
    # Exact values for the lever arms.
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (sp.nsimplify(fx, rational=True), sp.nsimplify(fy, rational=True))

    state.nodes = {
        nid: (sp.nsimplify(x, rational=True), sp.nsimplify(y, rational=True))
        for nid, (x, y) in state.nodes.items()
    }
    unknown_cut_ids = [eid for eid in cut_element_ids if eid in state.active_element_ids]
    if len(unknown_cut_ids) == 0 or len(unknown_cut_ids) > 3:
        return False

    # A valid cut leaves exactly two connected pieces (found by breadth-first search).
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

    # A side is solvable only if all its support reactions are known.
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

    # Moment center: any node of the chosen sub-truss.
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

        # local_node is the end of the member inside the sub-truss.
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

    active_eqs = []
    for eq in [sum_fx, sum_fy, sum_mz]:
        # nsimplify turns float noise into 0 and decimals into exact rationals.
        clean_eq = sp.nsimplify(eq, tolerance=1e-9, rational=True)
        if clean_eq != 0:
            active_eqs.append(clean_eq)

    solution = sp.linsolve(active_eqs, unknown_vars)

    if not solution:
        return False

    solution_vals = list(solution)[0]

    results = {}
    for eid, val in zip(unknown_cut_ids, solution_vals):
        simplified_val = sp.simplify(val)
        state.known_forces[eid] = simplified_val
        state.active_element_ids.remove(eid)
        results[eid] = simplified_val

    # Save the cut for the FBD drawing code (draw_FBDs).
    state.mos_cut_elements = cut_element_ids

    # Any node of the kept sub-truss anchors the FBD view.
    state.mos_target_node = list(sub_truss_nodes)[0]

    state.log_step(
        step_type="Method of Sections",
        target=f"Cut: {', '.join(map(str, cut_element_ids))}",
        results=results,
        explanation=f"Evaluated FBD of sub-truss nodes {sorted(list(sub_truss_nodes))} using Sum Fx, Sum Fy, and Sum Mz."
    )

    return True


def autonomous_moj_macro(state, include_collinear_load_rule=False):
    """
    Solves the whole truss by the method of joints: reactions, zero-force members, then joint by joint.

    Args:
        state (TrussState): Truss state, updated in place.
        include_collinear_load_rule (bool): Whether to apply rule 3 (collinear load at a joint).

    Returns:
        bool: True if every member was solved, False if the truss is internally indeterminate.
    """
    for nid, (fx, fy) in state.loads.items():
        state.loads[nid] = (
            sp.nsimplify(fx, rational=True),
            sp.nsimplify(fy, rational=True)
        )

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

    identify_zfms_by_inspection_tool(state, include_collinear_load_rule)

    made_progress = True

    while made_progress and state.active_element_ids:
        made_progress = False

        # Sorted for a reproducible solving order.
        for node_id in sorted(state.nodes.keys(), key=str):

            if solve_single_joint_tool(state, node_id):
                made_progress = True

                # A joint solve can expose new zero-force members, so restart from the first
                # node to prefer simpler joints.
                identify_zfms_by_inspection_tool(state, include_collinear_load_rule)
                break

    return len(state.active_element_ids) == 0


def programmatic_solution(solution_recipe, sampled_vars):
    """
    Runs a solution recipe in key order: the first step builds the TrussState, the middle
    steps solve it, and the last step extracts the answer. Raises RuntimeError if a middle
    step returns False.

    Args:
        solution_recipe (dict): Numbered steps, each {"function": callable, "inputs": {kwargs}}:
            {
                1: {"function": callable, "inputs": {static_kwargs}},
                2: {"function": callable, "inputs": {static_kwargs}},
                ...
            }
        sampled_vars (dict): Values for the problem's symbolic variables (e.g., {"P": 50.0}).

    Returns:
        dict: The extracted solution, in the template's expected output schema.
    """
    sorted_keys = sorted(solution_recipe.keys())
    steps = [solution_recipe[k] for k in sorted_keys]

    init_step = steps[0]
    init_func = init_step["function"]
    init_inputs = init_step["inputs"]

    state = init_func(sampled_vars=sampled_vars, **init_inputs)

    solver_steps = steps[1:-1]

    for current_step in solver_steps:
        solver_func = current_step["function"]
        solver_inputs = current_step["inputs"]

        success = solver_func(state, **solver_inputs)

        if success is False:
            raise RuntimeError(f"Pipeline failed at step: {solver_func.__name__}")

    ext_step = steps[-1]
    ext_func = ext_step["function"]
    ext_inputs = ext_step["inputs"]

    ground_truth = ext_func(state, **ext_inputs)

    return ground_truth


def find_controlling_members(coeffs, limit_tension, limit_compression, offsets=None, rel_tol=1e-6):
    """
    Finds the largest load before any member reaches its tension or
    compression limit, and the members tied for that limit.

    Args:
        coeffs (dict): Member ID -> force per unit load (signed).
        limit_tension (float): Tension limit for every member.
        limit_compression (float): Compression limit for every member (positive).
        offsets (dict, optional): Member ID -> force at zero load. Defaults to 0.
        rel_tol (float): Relative tolerance for treating two loads as tied.

    Returns:
        tuple: (load, state, members) with the governing load, "tension" or
            "compression", and the sorted list of tied member IDs.
    """
    offsets = offsets or {}

    # A member over its limit at zero load fails regardless of the load.
    already_failed = {}
    for eid in coeffs:
        offset = offsets.get(eid, 0.0)
        if offset > limit_tension:
            already_failed[eid] = "tension"
        elif offset < -limit_compression:
            already_failed[eid] = "compression"

    if already_failed:
        # Rank by how far each member is over its own limit.
        overage = {}
        for eid, state in already_failed.items():
            limit = limit_tension if state == "tension" else limit_compression
            overage[eid] = abs(offsets.get(eid, 0.0)) / limit

        worst_overage = max(overage.values())
        controlling_members = sorted(
            eid for eid, o in overage.items() if math.isclose(o, worst_overage, rel_tol=rel_tol)
        )
        controlling_state = already_failed[controlling_members[0]]
        controlling_load = 0.0
    else:
        # Load at which each member reaches its own limit.
        failure_loads = {}
        for eid, coeff in coeffs.items():
            offset = offsets.get(eid, 0.0)
            if coeff > 1e-9:
                load = (limit_tension - offset) / coeff
                if load > 0:
                    failure_loads[eid] = ("tension", load)
            elif coeff < -1e-9:
                load = (limit_compression + offset) / (-coeff)
                if load > 0:
                    failure_loads[eid] = ("compression", load)

        if not failure_loads:
            raise ValueError("No member forces bound the load in either tension or compression.")

        controlling_load = min(load for _, load in failure_loads.values())
        controlling_members = sorted(
            eid for eid, (_, load) in failure_loads.items()
            if math.isclose(load, controlling_load, rel_tol=rel_tol)
        )
        controlling_state = failure_loads[controlling_members[0]][0]

    # Members held at the governing limit by their offset alone also tie.
    limit_at_state = limit_tension if controlling_state == "tension" else -limit_compression
    for eid, coeff in coeffs.items():
        if eid in controlling_members:
            continue
        force_at_load = offsets.get(eid, 0.0) + coeff * controlling_load
        if math.isclose(force_at_load, limit_at_state, rel_tol=rel_tol, abs_tol=1e-6):
            controlling_members.append(eid)
    controlling_members = sorted(controlling_members)

    return controlling_load, controlling_state, controlling_members
