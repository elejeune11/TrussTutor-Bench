import pytest
import sympy as sp
from trusstutor_bench.truss_state import TrussState
from trusstutor_bench.statics_tools import solve_global_reactions_tool
from trusstutor_bench.statics_tools import identify_zfms_by_inspection_tool
from trusstutor_bench.statics_tools import solve_single_joint_tool
from trusstutor_bench.statics_tools import solve_method_of_sections_tool
from trusstutor_bench.statics_tools import autonomous_moj_macro
from trusstutor_bench.direct_stiffness_method import solve_truss
from trusstutor_bench.statics_tools import programmatic_solution


def test_solve_global_reactions_tool():
    """
    Verifies that the tool correctly solves for supports, mutates the state, 
    and appends the correct pedagogical trace.
    """
    L, P = sp.symbols('L P', positive=True)
    
    # Simple beam/truss layout: Pin at (0,0), Roller at (L,0), downward load P at L/2
    nodes = {
        1: (0, 0),
        2: (L, 0),
        3: (L/2, 5) 
    }
    elements = [] # Elements don't matter for rigid body global statics
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (0, -P)}
    
    # Initialize the state object
    state = TrussState(nodes, elements, supports, loads)
    
    # Run the atomic tool
    solve_global_reactions_tool(state)
    
    # 1. Verify State Mutation
    assert len(state.global_reactions) == 3
    assert state.global_reactions['R_1_x'] == 0
    assert state.global_reactions['R_1_y'] == P / 2
    assert state.global_reactions['R_2_y'] == P / 2
    
    # 2. Verify Pedagogical Logging
    assert len(state.trace) == 1
    assert state.trace[0]['step_type'] == "Global Statics"
    assert state.trace[0]['target'] == "Entire Truss"
    
    # Ensure variables were correctly cast to strings for the markdown log
    assert "P/2" in state.trace[0]['results']['R_1_y']


def test_global_reactions_indeterminate():
    """Verifies that the tool rejects globally indeterminate structures."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = []
    # Two pins = 4 unknown reactions (Indeterminate in 2D)
    supports = {1: ['x', 'y'], 2: ['x', 'y']} 
    loads = {2: (0, -100)}
    
    state = TrussState(nodes, elements, supports, loads)
    
    with pytest.raises(ValueError, match="globally statically indeterminate"):
        solve_global_reactions_tool(state)


def test_global_reactions_unstable_system():
    """Verifies that the tool rejects unstable/inconsistent boundary conditions."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = []
    # Two vertical rollers cannot resist a horizontal load
    supports = {1: ['y'], 2: ['y']} 
    loads = {2: (100, -100)} # 100 in the x-direction
    
    state = TrussState(nodes, elements, supports, loads)
    
    with pytest.raises(ValueError, match="singular or inconsistent"):
        solve_global_reactions_tool(state)


def test_global_reactions_no_supports():
    """Verifies that the tool safely bypasses structures with no supports."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = []
    supports = {} # Floating in space
    loads = {2: (0, -100)}
    
    state = TrussState(nodes, elements, supports, loads)
    solve_global_reactions_tool(state)
    
    # Ensure it exited cleanly without mutating the state or crashing
    assert len(state.global_reactions) == 0
    assert len(state.trace) == 0


def test_global_reactions_zero_load():
    """Verifies that an unloaded determinate structure correctly yields 0.0 reactions."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = []
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {} # No external loads applied
    
    state = TrussState(nodes, elements, supports, loads)
    solve_global_reactions_tool(state)
    
    assert float(state.global_reactions['R_1_x']) == 0.0
    assert float(state.global_reactions['R_1_y']) == 0.0
    assert float(state.global_reactions['R_2_y']) == 0.0
    assert len(state.trace) == 1


def test_identify_zfms_tool():
    """Verifies that the ZFM tool correctly identifies Rules 1 and 2."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (10, 10), 4: (20, 10), 5: (20, 0)}
    supports = {1: ['x', 'y'], 5: ['y']}
    loads = {3: (1000.0, -1000.0)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (2, 5)},
        {'id': 'C', 'nodes': (2, 3)},  # Vertical at T-Joint (ZFM)
        {'id': 'D', 'nodes': (1, 3)},
        {'id': 'E', 'nodes': (3, 4)},  # Top to corner (ZFM)
        {'id': 'F', 'nodes': (4, 5)}   # Right to corner (ZFM)
    ]
    
    state = TrussState(nodes, elements, supports, loads)
    identify_zfms_by_inspection_tool(state)
    
    assert state.known_forces['C'] == 0.0
    assert state.known_forces['E'] == 0.0
    assert state.known_forces['F'] == 0.0
    assert 'C' not in state.active_element_ids
    assert len(state.trace) == 2


def test_identify_zfms_rule_3():
    """Verifies ZFM Rule 3: Collinear load at a 2-member joint."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (10, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)}, # Horizontal
        {'id': 'B', 'nodes': (2, 3)}  # Vertical
    ]
    supports = {1: ['x', 'y']}
    loads = {2: (500.0, 0.0)} # Horizontal load collinear with A

    state = TrussState(nodes, elements, supports, loads)
    identify_zfms_by_inspection_tool(state, include_collinear_load_rule=True)
    
    assert state.known_forces['B'] == 0.0
    assert 'Rule 3' in state.trace[0]['step_type']


def test_identify_zfms_no_zfms():
    """Verifies the tool safely exits when no ZFMs exist."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 5)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (2, 3)},
        {'id': 'C', 'nodes': (1, 3)}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (0, -100)}
    
    state = TrussState(nodes, elements, supports, loads)
    identify_zfms_by_inspection_tool(state)
    
    assert len(state.known_forces) == 0
    assert len(state.trace) == 0


def test_identify_zfms_rule_2_alt_order():
    """Hits the alternate 'elif' branches for T-Joint collinearity (Lines 139-140)."""
    # T-Joint at Node 1. Collinear members are horizontal.
    nodes = {1: (0, 0), 2: (10, 0), 3: (-10, 0), 4: (0, 10)}
    
    # We purposefully place the non-collinear vertical member FIRST in the list.
    # This forces the tool to check the combination of el2 and el3.
    elements = [
        {'id': 'vert', 'nodes': (1, 4)},
        {'id': 'horiz1', 'nodes': (1, 2)},
        {'id': 'horiz2', 'nodes': (1, 3)}
    ]
    supports = {2: ['x', 'y'], 3: ['x', 'y'], 4: ['x', 'y']}
    loads = {} # Unloaded T-Joint
    
    state = TrussState(nodes, elements, supports, loads)
    identify_zfms_by_inspection_tool(state)
    
    assert state.known_forces['vert'] == 0.0


def test_identify_zfms_rule_3_alt_order():
    """Hits the alternate 'elif' branch for loaded joints (Lines 170-171)."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (0, 10)}
    
    # We purposefully place the vertical member FIRST.
    elements = [
        {'id': 'A', 'nodes': (1, 3)}, # Vertical (el1)
        {'id': 'B', 'nodes': (1, 2)}  # Horizontal (el2)
    ]
    supports = {2: ['x', 'y'], 3: ['x', 'y']}
    
    # Horizontal load collinear with element B (which is now el2)
    loads = {1: (500.0, 0.0)}

    state = TrussState(nodes, elements, supports, loads)
    identify_zfms_by_inspection_tool(state, include_collinear_load_rule=True)
    
    assert state.known_forces['A'] == 0.0


def test_solve_single_joint_tool():
    """Verifies that the tool correctly solves a standard 2-unknown joint."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (0, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)}, # Horizontal
        {'id': 'B', 'nodes': (1, 3)}  # Vertical
    ]
    supports = {2: ['x', 'y'], 3: ['x', 'y']}
    # Apply a diagonal load at node 1
    loads = {1: (100.0, 200.0)} 
    
    state = TrussState(nodes, elements, supports, loads)
    
    # Try to solve Node 1 (has 2 unknowns)
    success = solve_single_joint_tool(state, 1)
    
    assert success is True
    # To balance a +100 external load, member A must pull with -100
    assert float(state.known_forces['A']) == -100.0
    # To balance a +200 external load, member B must pull with -200
    assert float(state.known_forces['B']) == -200.0
    
    assert 'A' not in state.active_element_ids
    assert 'B' not in state.active_element_ids
    assert state.trace[0]['target'] == 1
    assert "Decoupled System" in state.trace[0]['explanation']


def test_solve_single_joint_skip_too_many_unknowns():
    """Verifies the tool safely bypasses joints with > 2 unknowns."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (0, 10), 4: (10, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (1, 3)},
        {'id': 'C', 'nodes': (1, 4)}
    ]
    supports = {}
    loads = {}
    
    state = TrussState(nodes, elements, supports, loads)
    success = solve_single_joint_tool(state, 1)
    
    # Should return False and leave state untouched
    assert success is False
    assert len(state.known_forces) == 0


def test_solve_single_joint_with_prior_knowledge():
    """Verifies the tool incorporates reactions and already-solved members."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = [{'id': 'A', 'nodes': (1, 2)}]
    supports = {1: ['x', 'y']}
    loads = {}
    
    state = TrussState(nodes, elements, supports, loads)
    
    # Manually inject prior knowledge
    state.global_reactions['R_1_x'] = -500.0
    state.global_reactions['R_1_y'] = 0.0
    
    # Solve Node 1 (1 unknown: member A)
    success = solve_single_joint_tool(state, 1)
    
    assert success is True
    # If reaction pushes left with 500, member A must pull right (tension) with +500
    assert float(state.known_forces['A']) == 500.0
    assert "Trivial System" in state.trace[0]['explanation']


def test_solve_single_joint_known_members_and_y_reaction():
    """Hits the R_y reaction branch and the known_elements loop (Lines 227-232)."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (0, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)}, # Unknown (Active)
        {'id': 'B', 'nodes': (1, 3)}  # Known!
    ]
    supports = {1: ['y']} 
    loads = {}
    
    state = TrussState(nodes, elements, supports, loads)
    
    # Inject prior knowledge directly into the state
    state.global_reactions['R_1_y'] = 500.0
    state.known_forces['B'] = -500.0  # Compression, pushing down with 500
    state.active_element_ids.remove('B') # Remove from active pool to mark as known
    
    # Solve. Y forces: R_1_y (up 500) + B (down 500) = 0. X forces: A = 0.
    success = solve_single_joint_tool(state, 1)
    
    assert success is True
    assert float(state.known_forces['A']) == 0.0


def test_solve_single_joint_singular_matrix():
    """Hits the 'if not solution' singular matrix branch (Line 251)."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (-10, 0)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)}, # Horizontal Right
        {'id': 'B', 'nodes': (1, 3)}  # Horizontal Left
    ]
    supports = {}
    
    # Apply a purely vertical load to a joint with only horizontal members.
    # This creates an impossible mathematical state (0 = 100).
    loads = {1: (0, 100.0)} 
    
    state = TrussState(nodes, elements, supports, loads)
    
    # System is geometrically inconsistent
    success = solve_single_joint_tool(state, 1)
    
    assert success is False


def test_method_of_sections_standard_cut():
    """Verifies a standard 3-member vertical cut, choosing the side with knowns."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (10, 10), 4: (0, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)}, # Bottom
        {'id': 'B', 'nodes': (2, 3)}, # Right Vertical
        {'id': 'C', 'nodes': (3, 4)}, # Top
        {'id': 'D', 'nodes': (4, 1)}, # Left Vertical
        {'id': 'E', 'nodes': (1, 3)}  # Diagonal
    ]
    # Left side has supports, right side is floating
    supports = {1: ['x', 'y'], 4: ['x']} 
    loads = {2: (500.0, 0.0)} # Pulling bottom-right node to the right
    
    state = TrussState(nodes, elements, supports, loads)
    
    # We purposefully haven't solved global reactions yet.
    # Therefore, the tool MUST choose the right-hand side (Nodes 2 & 3)
    # Cut Top (C), Diagonal (E), and Bottom (A)
    success = solve_method_of_sections_tool(state, ['C', 'E', 'A'])
    
    assert success is True
    # Evaluated right side: SumFx = 0 means member A and E must pull back.
    # With geometry, it should fully solve the 3 members.
    assert 'A' in state.known_forces
    assert 'C' in state.known_forces
    assert 'E' in state.known_forces
    assert len(state.active_element_ids) == 2 # Only B and D are left
    assert "Method of Sections" in state.trace[0]['step_type']


def test_method_of_sections_too_many_unknowns():
    """Verifies the tool rejects cuts with > 3 unknowns."""
    nodes = {1: (0,0), 2: (10,0)} # Topology doesn't matter for this check
    elements = [{'id': 'A', 'nodes': (1,2)}]
    state = TrussState(nodes, elements, {}, {})
    # Fake a state to trigger the length check
    state.active_element_ids = {'A', 'B', 'C', 'D'} 
    
    success = solve_method_of_sections_tool(state, ['A', 'B', 'C', 'D'])
    assert success is False


def test_method_of_sections_invalid_graph_split():
    """Verifies the tool rejects a cut that doesn't split the truss in two."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 10)}
    elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (2, 3)},
        {'id': 'C', 'nodes': (1, 3)}
    ]
    state = TrussState(nodes, elements, {}, {})
    
    # Cutting only one member of a triangle leaves it as 1 connected component
    success = solve_method_of_sections_tool(state, ['A'])
    assert success is False


def test_method_of_sections_missing_reactions():
    """Verifies the tool fails gracefully if both sides lack solved reactions."""
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = [{'id': 'A', 'nodes': (1, 2)}]
    # Both sides have unknown supports
    supports = {1: ['x', 'y'], 2: ['y']} 
    state = TrussState(nodes, elements, supports, {})
    
    success = solve_method_of_sections_tool(state, ['A'])
    
    # Fails because neither sub-truss has its supports solved yet
    assert success is False


def test_method_of_sections_with_reactions_and_knowns():
    """Hits reaction loops, known cut members, and reversed node order."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (10, 10), 4: (0, 10)}
    elements = [
        # Reversed node order: n2 (1) is in the left sub-truss, n1 (2) is out
        {'id': 'A', 'nodes': (2, 1)}, 
        {'id': 'B', 'nodes': (2, 3)},
        {'id': 'C', 'nodes': (3, 4)},
        {'id': 'D', 'nodes': (4, 1)},
        {'id': 'E', 'nodes': (1, 3)}
    ]
    supports = {1: ['x', 'y'], 4: ['x']}
    loads = {}
    
    state = TrussState(nodes, elements, supports, loads)
    
    # Inject physically CONSISTENT reactions so SymPy doesn't reject the matrix
    # If C pulls with 500, R_4_x must push back with -500 to balance the moment
    state.global_reactions['R_4_x'] = -500.0
    state.global_reactions['R_1_y'] = 0.0
    state.global_reactions['R_1_x'] = -100.0
    
    # Manually mark one of our cut members as ALREADY solved
    state.known_forces['C'] = 500.0
    state.active_element_ids.remove('C')
    
    # Cut A, C, and E. 
    success = solve_method_of_sections_tool(state, ['A', 'C', 'E'])
    
    assert success is True
    assert 'A' in state.known_forces
    assert 'E' in state.known_forces


def test_method_of_sections_singular_matrix():
    """Hits the singular matrix/inconsistent physics branch."""
    # Two rectangular panels. Left side: 1, 2, 3. Right side: 4, 5, 6.
    nodes = {1: (0,0), 2: (0,5), 3: (0,10), 4: (10,0), 5: (10,5), 6: (10,10)}
    
    # Three perfectly horizontal members connecting the two halves
    elements = [
        {'id': 'Top', 'nodes': (3, 6)},
        {'id': 'Mid', 'nodes': (2, 5)},
        {'id': 'Bot', 'nodes': (1, 4)},
        {'id': 'Left1', 'nodes': (1, 2)},
        {'id': 'Left2', 'nodes': (2, 3)},
        {'id': 'Right1', 'nodes': (4, 5)},
        {'id': 'Right2', 'nodes': (5, 6)}
    ]
    supports = {}
    
    # Apply a vertical load to the left sub-truss.
    # The three horizontal cut members CANNOT physically resist a vertical load.
    loads = {3: (0, 100.0)} 
    
    state = TrussState(nodes, elements, supports, loads)
    
    # The math system yields 100 = 0, meaning it is singular/inconsistent.
    success = solve_method_of_sections_tool(state, ['Top', 'Mid', 'Bot'])
    
    assert success is False


def test_method_of_sections_internal_cut_member():
    """Hits the boundary 'else' branch (Line 373) for a completely internal cut member."""
    # Build two triangles connected by two horizontal bridges
    nodes = {
        1: (0, 0), 2: (0, 10), 3: (5, 5),   # Left sub-truss
        4: (10, 0), 5: (10, 10), 6: (15, 5) # Right sub-truss
    }
    elements = [
        # Left Triangle
        {'id': 'L1', 'nodes': (1, 2)},
        {'id': 'L2', 'nodes': (2, 3)},
        {'id': 'L3', 'nodes': (1, 3)},
        # Right Triangle
        {'id': 'R1', 'nodes': (4, 5)},
        {'id': 'R2', 'nodes': (5, 6)},
        {'id': 'R3', 'nodes': (4, 6)},
        # Bridges connecting the two halves
        {'id': 'Top', 'nodes': (2, 5)},
        {'id': 'Bot', 'nodes': (1, 4)}
    ]
    supports = {1: ['x', 'y'], 2: ['x']}
    loads = {}
    state = TrussState(nodes, elements, supports, loads)
    
    # Inject known reactions so the left side is "solvable"
    state.global_reactions = {'R_1_x': 0.0, 'R_1_y': 0.0, 'R_2_x': 0.0}
    
    # We cut the two bridges, but ALSO cut L3. 
    # Because L1 and L2 keep the left side connected, the graph still splits in two.
    # However, L3 is entirely contained within the left side, triggering the else block.
    success = solve_method_of_sections_tool(state, ['Top', 'Bot', 'L3'])
    
    assert success is False


def test_autonomous_moj_macro_complete_solve():
    """Verifies the macro can solve a truss end-to-end without float contradictions."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 10)}
    elements = [
        {'id': 'Bottom', 'nodes': (1, 2)},
        {'id': 'Left', 'nodes': (1, 3)},
        {'id': 'Right', 'nodes': (2, 3)}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (100.0, -1000.0)}
    
    state = TrussState(nodes, elements, supports, loads)
    success = autonomous_moj_macro(state)
    
    assert success is True
    # Verify the active pool is completely empty
    assert len(state.active_element_ids) == 0
    # Verify all 3 members have known forces
    assert len(state.known_forces) == 3
    # Verify global reactions were calculated
    assert 'R_1_x' in state.global_reactions


def test_autonomous_moj_macro_internally_indeterminate():
    """Verifies the macro safely exits if it cannot solve the full truss."""
    # A square panel with an "X" cross-brace (1 degree internally indeterminate)
    nodes = {1: (0, 0), 2: (10, 0), 3: (10, 10), 4: (0, 10)}
    elements = [
        {'id': 'Bottom', 'nodes': (1, 2)}, 
        {'id': 'Right', 'nodes': (2, 3)},
        {'id': 'Top', 'nodes': (3, 4)}, 
        {'id': 'Left', 'nodes': (4, 1)},
        {'id': 'Diag1', 'nodes': (1, 3)}, # The X brace
        {'id': 'Diag2', 'nodes': (2, 4)}  
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {4: (100.0, 0)}
    
    state = TrussState(nodes, elements, supports, loads)
    success = autonomous_moj_macro(state)
    
    assert success is False
    # It should successfully solve global reactions, but get permanently stuck 
    # trying to solve the joints because every joint has at least 3 unknowns.
    assert len(state.active_element_ids) > 0


def test_autonomous_moj_macro_global_reaction_failure():
    """Hits the ValueError exception block (Lines 427-434) in the macro."""
    # Simple geometry, but with too many supports (Two pins = 4 unknowns)
    nodes = {1: (0, 0), 2: (10, 0)}
    elements = [{'id': 'A', 'nodes': (1, 2)}]
    supports = {1: ['x', 'y'], 2: ['x', 'y']} 
    loads = {2: (0, -100.0)}
    
    state = TrussState(nodes, elements, supports, loads)
    
    # The macro should catch the ValueError, log the error, and return False
    success = autonomous_moj_macro(state)
    
    assert success is False
    assert len(state.trace) == 1
    assert state.trace[0]['step_type'] == "Error"
    assert "Global Reactions" in state.trace[0]['target']

#######################################################################
#######################################################################


def helper_assert_close(val1, val2, tol=1e-5):
    """Helper to handle type casting from SymPy expressions to float comparison."""
    assert abs(float(val1) - float(val2)) < tol


def extract_dsm_reactions(nodes, supports, rx_array):
    """Maps the flat 2N DSM reaction array into a pedagogical string-keyed dictionary."""
    reactions_dict = {}
    node_list = list(nodes.keys()) # Assumes array ordering matches dict insertion order
    
    for i, node_id in enumerate(node_list):
        if node_id in supports:
            if 'x' in supports[node_id]:
                reactions_dict[f"R_{node_id}_x"] = rx_array[2 * i]
            if 'y' in supports[node_id]:
                reactions_dict[f"R_{node_id}_y"] = rx_array[2 * i + 1]
                
    return reactions_dict

#######################################################################
#######################################################################


def test_dsm_vs_moj_simple_triangle():
    """Validates the autonomous Method of Joints macro against DSM for a simple triangle."""
    # Using the exact 3-member geometry we previously verified passes the SymPy float tolerance
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (5.0, 10.0)}
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'Bottom', 'nodes': (1, 2), **e_props},
        {'id': 'Left', 'nodes': (1, 3), **e_props},
        {'id': 'Right', 'nodes': (2, 3), **e_props}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (100.0, -1000.0)}

    # 1. Solve using the Direct Stiffness Method (Ground Truth)
    _, dsm_forces, dsm_rx_array = solve_truss(nodes, elements, supports, loads)
    dsm_reactions = extract_dsm_reactions(nodes, supports, dsm_rx_array)

    # 2. Solve using the Pedagogical Statics Framework
    state = TrussState(nodes, elements, supports, loads)
    success = autonomous_moj_macro(state)
    
    assert success is True

    # 3. Cross-validate global reactions
    for rx_key, r_val in dsm_reactions.items():
        assert rx_key in state.global_reactions
        helper_assert_close(state.global_reactions[rx_key], r_val)

    # 4. Cross-validate internal member forces
    for eid, f_val in dsm_forces.items():
        assert eid in state.known_forces
        helper_assert_close(state.known_forces[eid], f_val)


def test_dsm_vs_zfms_by_inspection():
    """Verifies that members identified as ZFMs by inspection have 0.0 force in DSM."""
    nodes = {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (10.0, 10.0), 4: (20.0, 0.0)}
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'A', 'nodes': (1, 2), **e_props},
        {'id': 'B', 'nodes': (2, 4), **e_props},
        {'id': 'C', 'nodes': (2, 3), **e_props},  # Vertical ZFM at T-joint
        {'id': 'D', 'nodes': (1, 3), **e_props},
        {'id': 'E', 'nodes': (3, 4), **e_props}   # <-- Stabilizing diagonal
    ]
    supports = {1: ['x', 'y'], 4: ['y']}
    loads = {3: (500.0, -500.0)}

    # DSM solution
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # Statics framework sweep
    state = TrussState(nodes, elements, supports, loads)
    autonomous_moj_macro(state)

    # Both frameworks must agree that member C carries absolutely no load
    helper_assert_close(state.known_forces['C'], 0.0)
    helper_assert_close(dsm_forces['C'], 0.0)


def test_dsm_vs_method_of_sections():
    """Validates a targeted Method of Sections cut directly against the DSM solution."""
    nodes = {
        1: (0.0, 0.0), 2: (10.0, 0.0), 3: (20.0, 0.0),
        4: (5.0, 10.0), 5: (15.0, 10.0)
    }
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'L1', 'nodes': (1, 2), **e_props}, {'id': 'L2', 'nodes': (2, 3), **e_props},
        {'id': 'U1', 'nodes': (4, 5), **e_props},
        {'id': 'D1', 'nodes': (1, 4), **e_props}, {'id': 'D2', 'nodes': (2, 4), **e_props},
        {'id': 'D3', 'nodes': (2, 5), **e_props}, {'id': 'D4', 'nodes': (3, 5), **e_props}
    ]
    supports = {1: ['x', 'y'], 3: ['y']}
    loads = {2: (0.0, -1000.0), 5: (0.0, -1000.0)}

    # DSM solution
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # Initialize statics state and pre-solve global reactions
    state = TrussState(nodes, elements, supports, loads)
    solve_global_reactions_tool(state)

    # Perform a vertical cut through the middle panel
    cut_members = ['U1', 'D3', 'L2']
    success = solve_method_of_sections_tool(state, cut_members)

    assert success is True

    # Assert that isolated section equilibrium exactly matches global stiffness matrix
    for eid in cut_members:
        helper_assert_close(state.known_forces[eid], dsm_forces[eid])


def test_dsm_vs_moj_irrational_geometry():
    """
    Tests if the solver survives geometries that result in repeating decimals (e.g. 1/3).
    This forces the symbolic solver to handle irrational-looking floats.
    """
    # Placing a node at x=3.3333333333... forces the math to handle repeating decimals
    nodes = {1: (0, 0), 2: (10.0/3.0, 10.0), 3: (20.0/3.0, 0)}
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'A', 'nodes': (1, 2), **e_props},
        {'id': 'B', 'nodes': (2, 3), **e_props},
        {'id': 'C', 'nodes': (1, 3), **e_props}
    ]
    supports = {1: ['x', 'y'], 3: ['y']}
    loads = {2: (0, -100.0)}

    # 1. DSM Truth
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # 2. Hybrid Symbolic Macro
    state = TrussState(nodes, elements, supports, loads)
    # The nsimplify logic inside the macro should catch the float noise 
    # from 10/3 and snap it to perfect fractions.
    success = autonomous_moj_macro(state)
    
    assert success is True
    for eid, force in dsm_forces.items():
        helper_assert_close(state.known_forces[eid], force)


def test_dsm_vs_moj_parametric_symbolic_load():
    """
    Tests that we maintain symbolic capabilities by using a SymPy Symbol as a load.
    This proves we didn't break the parametric power while adding the float filter.
    """
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 10)}
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'A', 'nodes': (1, 2), **e_props},
        {'id': 'B', 'nodes': (1, 3), **e_props},
        {'id': 'C', 'nodes': (2, 3), **e_props}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    
    # Define a symbolic load P
    P = sp.Symbol('P')
    loads = {3: (0, -P)}
    
    state = TrussState(nodes, elements, supports, loads)
    # We cannot use DSM for symbolic loads, so we check if the 
    # autonomous_moj_macro returns a symbolic expression for the forces.
    success = autonomous_moj_macro(state)
    
    assert success is True
    # The solver should return forces in terms of P
    assert isinstance(state.known_forces['A'], sp.Expr)
    assert P in state.known_forces['A'].free_symbols


def test_dsm_vs_moj_trigonometric_geometry():
    """
    Tests the symbolic solver against a 30-60-90 triangle. 
    This forces the use of sqrt(3) symbolic constants in all moment arms 
    and force projections.
    """
    # Geometry: 30-60-90 triangle. Height = 5 * sqrt(3), Base = 5.
    # We use sp.sqrt(3) to keep it strictly symbolic.
    sqrt3 = sp.sqrt(3)
    nodes = {
        1: (0, 0),
        2: (5, 0),
        3: (2.5, 2.5 * sqrt3)
    }
    
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'B1', 'nodes': (1, 2), **e_props},
        {'id': 'D1', 'nodes': (1, 3), **e_props},
        {'id': 'D2', 'nodes': (2, 3), **e_props}
    ]
    # Pin at 1, Roller at 2
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (100.0, -500.0)}

    # 1. DSM Truth (Numerical)
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # 2. Hybrid Symbolic Macro
    # By passing symbolic nodes, we ensure the solver doesn't just treat this as a decimal
    state = TrussState(nodes, elements, supports, loads)
    success = autonomous_moj_macro(state)
    
    assert success is True
    
    # Check that we haven't lost symbolic precision
    for eid, force in dsm_forces.items():
        # Verify the symbolic result equals the DSM result within tolerance
        helper_assert_close(state.known_forces[eid], force)
        
        # Verify the result is stored as a SymPy expression (or float/int)
        # This confirms we didn't cast to float too early
        assert isinstance(state.known_forces[eid], (sp.Expr, float, int))

    # Verify that the internal solver logic actually used the symbolic constants
    # (Checking the structure of the force in member D1)
    force_d1 = state.known_forces['D1']
    assert any(isinstance(arg, sp.Pow) or 'sqrt' in str(arg) 
               for arg in sp.preorder_traversal(sp.sympify(force_d1)))


def test_dsm_vs_moj_leaky_irrational_geometry():
    """
    Tests if the solver survives inputs that are truncated approximations 
    of irrational numbers (e.g., 1.732 instead of sqrt(3)).
    """
    # 30-60-90 triangle with a "leaky" float approximation for sqrt(3)
    # 5 * 1.732 = 8.66
    nodes = {1: (0, 0), 2: (5, 0), 3: (2.5, 8.66)}
    
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [
        {'id': 'B1', 'nodes': (1, 2), **e_props},
        {'id': 'D1', 'nodes': (1, 3), **e_props},
        {'id': 'D2', 'nodes': (2, 3), **e_props}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (100.0, -500.0)}

    # 1. DSM Truth
    # DSM will treat 8.66 as exactly 8.66.
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # 2. Hybrid Symbolic Macro
    # The sanitization layer will call nsimplify(8.66), 
    # which will convert it to the fraction 433/50.
    state = TrussState(nodes, elements, supports, loads)
    success = autonomous_moj_macro(state)
    
    assert success is True
    
    # Validation
    for eid, force in dsm_forces.items():
        # Because we cleaned the input to exact fractions, the symbolic solver
        # will yield a result slightly different from the DSM's pure float result.
        # We use a slightly wider tolerance because 8.66 != sqrt(3)*5.
        assert abs(float(state.known_forces[eid]) - float(force)) < 0.1


def test_asymmetrical_roof_zfms():
    """
    Tests that the ZFM tool correctly identifies BH, DF, and CF as zero-force members
    in an asymmetrically loaded roof truss, proving the cascade logic and collinearity checks.
    """
    nodes = {
        "A": (0, 0), "B": (2.5, 0), "C": (5, 0), "D": (7.5, 0),
        "E": (10, 0), "F": (7.5, 2.5), "G": (5, 5), "H": (2.5, 2.5),
    }

    connectivity = [
        ("AB", "A", "B"), ("BC", "B", "C"), ("CD", "C", "D"), ("DE", "D", "E"),
        ("EF", "E", "F"), ("FG", "F", "G"), ("GH", "G", "H"), ("HA", "H", "A"),
        ("BH", "B", "H"), ("CH", "C", "H"), ("CG", "C", "G"), ("CF", "C", "F"),
        ("DF", "D", "F"),
    ]

    loads = {
        "A": (0.0, -300.0),
        "H": (0.0, -600.0),
        "E": (0.0, -300.0)
    }

    supports = {
        "A": ["y"],
        "E": ["x", "y"]
    }

    elements = [{'id': eid, 'nodes': (n1, n2)} for eid, n1, n2 in connectivity]
    state = TrussState(nodes, elements, supports, loads)
    
    # 1. Run the tool
    success = identify_zfms_by_inspection_tool(state)

    # 2. Assert the tool reported making progress
    assert success is True, "The tool should have found ZFMs and returned True."

    # 3. Assert exactly 3 members were solved (prevents over-eager zeroing)
    assert len(state.known_forces) == 3, f"Expected exactly 3 ZFMs, found {len(state.known_forces)}"

    # 4. Assert the specific members are correctly logged and removed from active pool
    expected_zfms = ["BH", "DF", "CF"]
    for zfm in expected_zfms:
        assert zfm in state.known_forces, f"Failed to identify {zfm} as a ZFM."
        assert float(state.known_forces[zfm]) == 0.0, f"Force in {zfm} should be exactly 0.0."
        assert zfm not in state.active_element_ids, f"{zfm} was not removed from active_element_ids."


def test_method_of_sections_vs_dsm_benchmark():
    """
    Benchmarks the exact symbolic Method of Sections solver against the 
    numerical Direct Stiffness Method (DSM) for a statically determinate roof truss.
    """
    # 1. Setup exact symbolic geometry
    theta = 20 * sp.pi / 180

    nodes = {
        0: (0, 0),
        1: (10, 0),
        2: (20, 0),
        3: (30, 0),
        4: (20, -10 * sp.tan(theta)),
        5: (10, -10 * sp.tan(theta))
    }

    connectivity = [
        ("A", 0, 1), ("B", 0, 5), ("C", 1, 5), ("D", 1, 2),
        ("E", 1, 4), ("F", 4, 5), ("G", 2, 4), ("H", 2, 3), ("I", 3, 4)
    ]

    loads = {
        1: (0.0, -60.0),
        2: (0.0, -80.0)
    }

    supports = {
        0: ['x', 'y'],
        3: ['y']
    }

    # Inject material properties exactly like the leaky geometry test
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [{'id': eid, 'nodes': (n1, n2), **e_props} for eid, n1, n2 in connectivity]
    
    # 2. Execute Exact Symbolic Solution
    state = TrussState(nodes, elements, supports, loads)
    solve_global_reactions_tool(state)
    
    cut_element_ids = ["D", "E", "F"]
    success = solve_method_of_sections_tool(state, cut_element_ids)
    
    assert success is True, "Method of sections failed to resolve the cut."
    
    for eid in cut_element_ids:
        assert eid in state.known_forces, f"Member {eid} was not found in known_forces."

    # 3. Execute Numerical DSM Solution
    _, dsm_forces, _ = solve_truss(nodes, elements, supports, loads)

    # 4. Benchmarking Assertions
    for eid in cut_element_ids:
        # Convert exact SymPy answer to decimal
        symbolic_force = float(state.known_forces[eid].evalf())
        dsm_force = dsm_forces[eid]
        
        # Since this uses exact SymPy geometry, the difference should be microscopic
        assert abs(symbolic_force - float(dsm_force)) < 1e-5, \
            f"Benchmark Mismatch on {eid}: Symbolic = {symbolic_force}, DSM = {dsm_force}"


def test_autonomous_moj_vs_dsm_parametric_benchmark():
    """
    Benchmarks the fully autonomous Method of Joints macro against the DSM.
    Demonstrates how to evaluate a parametric symbolic solution (using L) 
    against a specific numerical instance.
    """
    # 1. Setup exact symbolic geometry
    L = sp.Symbol('L', positive=True)
    nodes = {
        "a": (0, 0),
        "b": (L / 2, sp.sqrt(3) * L / 2),
        "c": (3 * L / 2, sp.sqrt(3) * L / 2),
        "d": (5 * L / 2, sp.sqrt(3) * L / 2),
        "e": (3 * L, 0),
        "f": (2 * L, 0),
        "g": (L, 0),
    }

    connectivity = [
        ("ab", "a", "b"), ("bc", "b", "c"), ("cd", "c", "d"),
        ("de", "d", "e"), ("ef", "e", "f"), ("fg", "f", "g"),
        ("ag", "a", "g"), ("bg", "b", "g"), ("cg", "c", "g"),
        ("cf", "c", "f"), ("df", "d", "f")
    ]

    loads = {
        "g": (0.0, -100.0),
        "f": (0.0, -50.0)
    }

    supports = {
        "a": ['x', 'y'],
        "e": ['y']
    }

    # Inject material properties for DSM compatibility
    e_props = {'E': 200e9, 'A': 0.01}
    elements = [{'id': eid, 'nodes': (n1, n2), **e_props} for eid, n1, n2 in connectivity]
    
    # 2. Execute Exact Symbolic Solution
    state = TrussState(nodes, elements, supports, loads)
    solve_global_reactions_tool(state)
    success = autonomous_moj_macro(state)
    
    assert success is True, "MoJ macro failed to resolve the truss."
    assert len(state.active_element_ids) == 0, "Not all elements were solved."

    # 3. Setup Numerical DSM Solution
    # We must choose a specific value for L to run the numerical solver
    L_test_value = 10.0
    
    # Convert symbolic nodes to floating-point numerical nodes for the DSM engine
    dsm_nodes = {
        nid: (
            float(sp.sympify(x).subs(L, L_test_value).evalf()), 
            float(sp.sympify(y).subs(L, L_test_value).evalf())
        )
        for nid, (x, y) in nodes.items()
    }
    
    # Execute DSM on the numerical geometry
    _, dsm_forces, _ = solve_truss(dsm_nodes, elements, supports, loads)

    # 4. Benchmarking Assertions
    for eid in dsm_forces.keys():
        # Pull the exact formula (e.g., -500*sqrt(3)/9)
        symbolic_expr = state.known_forces[eid]
        
        # Substitute L_test_value (in case L didn't cancel out) and convert to float
        symbolic_force_decimal = float(symbolic_expr.subs(L, L_test_value).evalf())
        dsm_force_decimal = dsm_forces[eid]
        
        # Compare with strict precision
        assert abs(symbolic_force_decimal - float(dsm_force_decimal)) < 1e-5, \
            f"Benchmark Mismatch on {eid}: Symbolic = {symbolic_force_decimal}, DSM = {dsm_force_decimal}"


def dummy_initializer(sampled_vars, static_val):
    """Acts as the creator. Returns a simple dictionary as the 'State'."""
    return {
        "sampled_input": sampled_vars.get("P"),
        "static_input": static_val,
        "execution_trace": []
    }

def dummy_mutator_success(state, step_name):
    """Acts as a solver that succeeds. Mutates the state and returns True."""
    state["execution_trace"].append(step_name)
    state["is_solved"] = True
    return True

def dummy_mutator_failure(state, step_name):
    """Acts as a solver that fails (e.g., singular matrix). Returns False."""
    state["execution_trace"].append(step_name)
    return False

def dummy_extractor(state, target_key):
    """Acts as the final step. Returns specific data from the state."""
    state["execution_trace"].append("extractor_ran")
    return {
        "extracted_value": state.get(target_key),
        "trace": state["execution_trace"]
    }


# =====================================================================
# PYTEST SUITE
# =====================================================================

def test_programmatic_solution_happy_path():
    """
    Verifies that the orchestrator successfully unpacks arguments, mutates 
    the state sequentially, and returns the expected ground truth.
    """
    recipe = {
        1: {"function": dummy_initializer, "inputs": {"static_val": "Node_A"}},
        2: {"function": dummy_mutator_success, "inputs": {"step_name": "global_reactions"}},
        3: {"function": dummy_mutator_success, "inputs": {"step_name": "method_of_joints"}},
        4: {"function": dummy_extractor, "inputs": {"target_key": "is_solved"}}
    }
    
    sampled_vars = {"P": 50.0}
    
    ground_truth = programmatic_solution(solution_recipe=recipe, sampled_vars=sampled_vars)
    
    # Assert the mutators successfully changed the state
    assert ground_truth["extracted_value"] is True
    
    # Assert chronological execution of the mutators and the extractor
    assert ground_truth["trace"] == ["global_reactions", "method_of_joints", "extractor_ran"]


def test_programmatic_solution_halts_on_failure():
    """
    Verifies that if a mutator returns False, the orchestrator immediately 
    halts and raises a RuntimeError with the correct function name.
    """
    recipe = {
        1: {"function": dummy_initializer, "inputs": {"static_val": "Node_A"}},
        2: {"function": dummy_mutator_success, "inputs": {"step_name": "step_1"}},
        3: {"function": dummy_mutator_failure, "inputs": {"step_name": "failing_step"}},
        4: {"function": dummy_extractor, "inputs": {"target_key": "is_solved"}}
    }
    
    sampled_vars = {"P": 50.0}
    
    # Assert that the specific RuntimeError is raised
    with pytest.raises(RuntimeError, match="Pipeline failed at step: dummy_mutator_failure"):
        programmatic_solution(solution_recipe=recipe, sampled_vars=sampled_vars)


def test_programmatic_solution_sorts_keys_correctly():
    """
    Verifies the orchestrator forces chronological execution based on integer 
    keys, even if the dictionary was constructed out of order.
    """
    # Keys are deliberately scrambled
    recipe = {
        99: {"function": dummy_extractor, "inputs": {"target_key": "static_input"}},
        1: {"function": dummy_initializer, "inputs": {"static_val": "Sorted_Correctly"}},
        50: {"function": dummy_mutator_success, "inputs": {"step_name": "middle_step"}}
    }
    
    sampled_vars = {"P": 50.0}
    
    ground_truth = programmatic_solution(solution_recipe=recipe, sampled_vars=sampled_vars)
    
    # If it didn't sort, it would have tried to run the extractor first and crashed.
    # We assert that the static input initialized in step 1 made it to step 99.
    assert ground_truth["extracted_value"] == "Sorted_Correctly"
    assert ground_truth["trace"] == ["middle_step", "extractor_ran"]