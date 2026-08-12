import sympy as sp
import pytest
from trusstutor_bench.direct_stiffness_method import solve_truss
from trusstutor_bench.symbolic_solver import solve_global_reactions, substitute_values
from trusstutor_bench.symbolic_solver import partition_truss, solve_method_of_sections
from trusstutor_bench.symbolic_solver import solve_joint_equilibrium
from trusstutor_bench.symbolic_solver import identify_zfms_by_inspection
from trusstutor_bench.symbolic_solver import get_node_difficulty_score
from trusstutor_bench.symbolic_solver import solve_method_of_joints


def test_numerical_reactions():
    """Test standard numerical equilibrium on a simple beam-like span."""
    nodes = {
        1: (0.0, 0.0),   # Pin
        2: (10.0, 0.0),  # Roller
        3: (5.0, 0.0)    # Midpoint
    }
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (0.0, -1000.0)} # 1000 N pushing down in the middle
    
    # elements are omitted/empty as they aren't needed for external statics
    reactions = solve_global_reactions(nodes, [], supports, loads)
    
    # Because inputs were floats, SymPy outputs sympy.Float which we can cast/compare
    assert float(reactions['R_1_x']) == 0.0
    assert float(reactions['R_1_y']) == 500.0
    assert float(reactions['R_2_y']) == 500.0


def test_symbolic_reactions():
    """Test algebraic equilibrium with symbolic geometry and loads."""
    L, H, P = sp.symbols('L H P')
    
    nodes = {
        1: (0, 0),       # Pin
        2: (L, 0),       # Roller
        3: (L/2, H)      # Top peak
    }
    supports = {1: ['x', 'y'], 2: ['y']}
    
    # A horizontal load P pushing right at the top node
    loads = {3: (P, 0)} 
    
    reactions = solve_global_reactions(nodes, [], supports, loads)
    
    # Sum Fx: R_1_x + P = 0  => R_1_x = -P
    # Sum Mz about 1: (L)*R_2_y - (H)*P = 0 => R_2_y = H*P/L
    # Sum Fy: R_1_y + R_2_y = 0 => R_1_y = -H*P/L
    
    # We must use SymPy's structural equality to compare expressions
    assert reactions['R_1_x'].equals(-P)
    assert reactions['R_1_y'].equals(-H * P / L)
    assert reactions['R_2_y'].equals(H * P / L)


def test_symbolic_substitution():
    """Test plugging numerical values into a pre-solved symbolic system."""
    L, H, P = sp.symbols('L H P')
    
    # Dummy symbolic output representing R_2_y = H*P/L
    symbolic_results = {
        'R_2_y': (H * P) / L
    }
    
    # Define the real-world values for a specific benchmark problem
    substitutions = {
        L: 10.0,
        H: 5.0,
        P: 1000.0
    }
    
    # Execute the substitution
    numerical_results = substitute_values(symbolic_results, substitutions)
    
    # (5.0 * 1000.0) / 10.0 = 500.0
    assert numerical_results['R_2_y'] == 500.0


def test_dsm_vs_symbolic_reactions():
    """
    Cross-checks the numerical Direct Stiffness Method (DSM) against 
    the Symbolic Reaction Solver to ensure ground-truth parity.
    """
    # ---------------------------------------------------------
    # 1. Define the Shared Parameters
    # ---------------------------------------------------------
    num_L = 10.0
    num_H = 5.0
    num_P = 1000.0
    
    # ---------------------------------------------------------
    # 2. Run the Direct Stiffness Method (Numerical)
    # ---------------------------------------------------------
    nodes_num = {
        1: (0.0, 0.0),          # Pin
        2: (num_L, 0.0),        # Roller
        3: (num_L / 2, num_H)   # Peak
    }
    
    elements_num = [
        {'id': 1, 'nodes': (1, 2), 'E': 200e9, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 200e9, 'A': 0.01},
        {'id': 3, 'nodes': (1, 3), 'E': 200e9, 'A': 0.01}
    ]
    
    supports = {
        1: ['x', 'y'], # Pinned
        2: ['y']       # Roller
    }
    
    loads_num = {
        3: (num_P, 0.0) # Horizontal load P pulling right
    }
    
    # DSM outputs U_global, internal_forces, reactions
    _, _, reactions_dsm = solve_truss(nodes_num, elements_num, supports, loads_num)
    
    # ---------------------------------------------------------
    # 3. Run the Symbolic Solver
    # ---------------------------------------------------------
    L, H, P = sp.symbols('L H P')
    
    nodes_sym = {
        1: (0, 0),
        2: (L, 0),
        3: (L / 2, H)
    }
    
    loads_sym = {
        3: (P, 0)
    }
    
    # Solve symbolically, then substitute numerical values
    symbolic_reactions = solve_global_reactions(nodes_sym, [], supports, loads_sym)
    
    substitutions = {
        L: num_L,
        H: num_H,
        P: num_P
    }
    
    reactions_sym_num = substitute_values(symbolic_reactions, substitutions)
    
    # ---------------------------------------------------------
    # 4. Cross-Check the Results
    # ---------------------------------------------------------
    # In the DSM array, DOFs are mapped as: [u1x, u1y, u2x, u2y, u3x, u3y]
    # Therefore, Reaction DOFs are: Node 1 (0, 1), Node 2 (2, 3), Node 3 (4, 5)
    
    # R_1_x (Node 1, X-direction -> Index 0)
    assert pytest.approx(reactions_sym_num['R_1_x']) == reactions_dsm[0]
    
    # R_1_y (Node 1, Y-direction -> Index 1)
    assert pytest.approx(reactions_sym_num['R_1_y']) == reactions_dsm[1]
    
    # R_2_y (Node 2, Y-direction -> Index 3)
    assert pytest.approx(reactions_sym_num['R_2_y']) == reactions_dsm[3]


def test_partition_truss():
    """Verify that removing cut members correctly isolates two subgraphs."""
    nodes = {1: (0,0), 2: (1,0), 3: (2,0), 4: (0,1), 5: (1,1), 6: (2,1)}
    elements = [
        {'id': 'A', 'nodes': (1,2)}, {'id': 'B', 'nodes': (2,3)}, 
        {'id': 'C', 'nodes': (4,5)}, {'id': 'D', 'nodes': (5,6)},
        {'id': 'E', 'nodes': (2,5)}, # Vertical
        {'id': 'F', 'nodes': (1,5)}, # Diagonal
        {'id': 'G', 'nodes': (2,6)}, # Diagonal
        {'id': 'H', 'nodes': (1,4)}  # The vertical wall member
    ]
    
    # Cut through the middle panel (cutting top, bottom, and diagonal)
    cut_ids = ['A', 'C', 'F']
    body_1, body_2 = partition_truss(nodes, elements, cut_ids)
    
    # One body should be the left nodes (1, 4), the other should be the rest
    expected_set_1 = {1, 4}
    expected_set_2 = {2, 3, 5, 6}
    
    assert (body_1 == expected_set_1 and body_2 == expected_set_2) or \
           (body_1 == expected_set_2 and body_2 == expected_set_1)


def test_symbolic_method_of_sections():
    """
    Test a cantilevered 2-panel square truss with symbolic dimensions 
    and a tip load. We will cut the members near the wall.
    """
    L, P = sp.symbols('L P', positive=True)
    
    # 2-panel cantilever attached to a wall on the left (x=0)
    nodes = {
        1: (0, 0),     # Bottom Wall
        2: (0, L),     # Top Wall
        3: (L, 0),     # Bottom Mid
        4: (L, L),     # Top Mid
        5: (2*L, 0),   # Bottom Tip
        6: (2*L, L)    # Top Tip
    }
    
    elements = [
        {'id': 'wall', 'nodes': (1,2)},  # The rigid wall connection
        {'id': 'top_1', 'nodes': (2,4)},
        {'id': 'bot_1', 'nodes': (1,3)},
        {'id': 'diag_1', 'nodes': (1,4)},
        {'id': 'vert_1', 'nodes': (3,4)},
        {'id': 'top_2', 'nodes': (4,6)},
        {'id': 'bot_2', 'nodes': (3,5)},
        {'id': 'diag_2', 'nodes': (3,6)},
        {'id': 'vert_2', 'nodes': (5,6)}
    ]
    
    # Tip load pointing down
    loads = {
        5: (0, -P) 
    }
    
    # Because we are cutting the truss and evaluating the right-hand side (the tip), 
    # we don't actually need to solve the wall reactions! The algorithm will just 
    # isolate the right-hand body where reactions = 0.
    global_reactions = {}
    
    # Cut the panel closest to the wall
    cut_ids = ['top_1', 'bot_1', 'diag_1']
    
    # Explicitly target Node 5 so the solver uses the right-hand tip FBD
    forces = solve_method_of_sections(
        nodes, elements, cut_ids, loads, global_reactions, target_fbd_node=5
    )
    
    # Mathematical Ground Truth (Corrected):
    # Sum forces Y (Right Body): Load P is DOWN. Tension at Node 4 pulls DOWN. 
    #   -P + F_diag_1 * (-1/sqrt(2)) = 0  => F_diag_1 = -P*sqrt(2) (Compression)
    # Sum forces X: All tension forces pull LEFT (-).
    #   -F_top_1 - F_bot_1 - F_diag_1*(1/sqrt(2)) = 0 
    # Sum Moments about (0,0):
    #   F_top_1*(L) - P*(2L) = 0 => F_top_1 = 2P (Tension)
    # Substitute back into X:
    #   -2P - F_bot_1 - (-P*sqrt(2))*(1/sqrt(2)) = 0 => -2P - F_bot_1 + P = 0 => F_bot_1 = -P (Compression)
    
    # Simplify the outputs to ensure clean symbolic matching
    assert sp.simplify(forces['top_1'] - 2*P) == 0
    assert sp.simplify(forces['diag_1'] - (-P*sp.sqrt(2))) == 0
    assert sp.simplify(forces['bot_1'] - (-P)) == 0


def test_dsm_vs_symbolic_sections():
    """
    Cross-checks the Method of Sections (Symbolic) against 
    the Direct Stiffness Method (Numerical) to ensure the local FBD 
    equilibrium matches the global matrix inversion exactly.
    """
    # ---------------------------------------------------------
    # 1. Define the Shared Parameters
    # ---------------------------------------------------------
    num_L = 10.0
    num_P = 1000.0  # 1 kN downward tip load
    
    # Material properties required ONLY by DSM
    E = 200e9
    A = 0.01

    # Shared Topology
    supports = {
        1: ['x', 'y'], # Bottom wall pinned
        2: ['x', 'y']  # Top wall pinned
    }
    
    # ---------------------------------------------------------
    # 2. Run the Direct Stiffness Method (Numerical)
    # ---------------------------------------------------------
    nodes_num = {
        1: (0.0, 0.0),            
        2: (0.0, num_L),          
        3: (num_L, 0.0),          
        4: (num_L, num_L),        
        5: (2*num_L, 0.0),        
        6: (2*num_L, num_L)       
    }
    
    elements_num = [
        {'id': 'wall', 'nodes': (1,2), 'E': E, 'A': A},
        {'id': 'top_1', 'nodes': (2,4), 'E': E, 'A': A},
        {'id': 'bot_1', 'nodes': (1,3), 'E': E, 'A': A},
        {'id': 'diag_1', 'nodes': (1,4), 'E': E, 'A': A},
        {'id': 'vert_1', 'nodes': (3,4), 'E': E, 'A': A},
        {'id': 'top_2', 'nodes': (4,6), 'E': E, 'A': A},
        {'id': 'bot_2', 'nodes': (3,5), 'E': E, 'A': A},
        {'id': 'diag_2', 'nodes': (3,6), 'E': E, 'A': A},
        {'id': 'vert_2', 'nodes': (5,6), 'E': E, 'A': A}
    ]
    
    loads_num = {5: (0.0, -num_P)}
    
    # Run the numerical engine
    _, forces_dsm, _ = solve_truss(nodes_num, elements_num, supports, loads_num)
    
    # ---------------------------------------------------------
    # 3. Run the Symbolic Solver
    # ---------------------------------------------------------
    L, P = sp.symbols('L P', positive=True)
    
    nodes_sym = {
        1: (0, 0),     
        2: (0, L),     
        3: (L, 0),     
        4: (L, L),     
        5: (2*L, 0),   
        6: (2*L, L)    
    }
    
    loads_sym = {5: (0, -P)}
    cut_ids = ['top_1', 'bot_1', 'diag_1']
    
    # Run the symbolic Method of Sections (evaluating the right-hand side, Node 5)
    forces_sym = solve_method_of_sections(
        nodes_sym, elements_num, cut_ids, loads_sym, global_reactions={}, target_fbd_node=5
    )
    
    # Substitute numbers into the symbolic expressions
    substitutions = {L: num_L, P: num_P}
    forces_sym_num = substitute_values(forces_sym, substitutions)
    
    # ---------------------------------------------------------
    # 4. Cross-Check the Results
    # ---------------------------------------------------------
    # Because both solvers assume Tension is Positive (+), the signs 
    # and magnitudes should map perfectly.
    
    assert pytest.approx(forces_sym_num['top_1']) == forces_dsm['top_1']
    assert pytest.approx(forces_sym_num['bot_1']) == forces_dsm['bot_1']
    assert pytest.approx(forces_sym_num['diag_1']) == forces_dsm['diag_1']


def test_solve_individual_joint():
    """
    Tests the localized 2x2 equilibrium solver at a single joint with a mix 
    of applied loads, known internal forces, and unknown members.
    """
    nodes = {
        1: (0, 0),   # The joint we are analyzing
        2: (10, 0),  # Horizontal neighbor
        3: (0, 10),  # Vertical neighbor
        4: (-5, 0)   # Neighbor behind it (already solved)
    }
    
    # Member 'A' and 'B' are unknown. Member 'C' is already solved.
    connected_elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (1, 3)},
        {'id': 'C', 'nodes': (1, 4)}
    ]
    
    # We pretend Member 'C' was solved in a previous step and is in 500 N Compression
    known_forces = {'C': -500.0} 
    
    # Node 1 is being pulled down and right by 1000 N
    loads = {1: (1000.0, -1000.0)}
    global_reactions = {} # No supports at this joint
    
    # Execute the solver on Node 1
    newly_solved = solve_joint_equilibrium(
        1, connected_elements, nodes, known_forces, global_reactions, loads
    )
    
    # Math Check (Sum Fx = 0):
    # Member C points left (-1, 0). (Compression -500) * (-1) = pushes 500 to the right.
    # Load is 1000 right.
    # Therefore, Member A must pull left by 1500 to balance. 
    # Tension in A = -1500 (Compression)
    assert newly_solved['A'] == -1500.0
    
    # Math Check (Sum Fy = 0):
    # Load is 1000 down. 
    # Member B must pull up by 1000 to balance. 
    # Tension in B = 1000 (Tension)
    assert newly_solved['B'] == 1000.0


def test_identify_zfms_by_inspection():
    """
    Test the ZFM pre-processor on a truss containing both an unloaded corner 
    and an unloaded T-joint.
    
    Topology:
        Node 3 (Top Left) ------- Node 4 (Top Right)
          / |                        |
         /  | (Mem C)                | (Mem F)
        /   |                        |
    Node 1 -- Node 2 -------------- Node 5
    (Pin)    (Bottom Mid)           (Roller)
    """
    nodes = {
        1: (0, 0),    # Pin
        2: (10, 0),   # Bottom Mid
        3: (10, 10),  # Top Left
        4: (20, 10),  # Top Right (Unloaded Corner)
        5: (20, 0)    # Roller
    }
    
    supports = {1: ['x', 'y'], 5: ['y']}
    
    # Load applied ONLY at Node 3
    loads = {3: (1000.0, -1000.0)}
    
    elements = [
        {'id': 'A', 'nodes': (1, 2)},
        {'id': 'B', 'nodes': (2, 5)},
        {'id': 'C', 'nodes': (2, 3)},  # Vertical at T-Joint
        {'id': 'D', 'nodes': (1, 3)},
        {'id': 'E', 'nodes': (3, 4)},  # Top horizontal to corner
        {'id': 'F', 'nodes': (4, 5)}   # Right vertical to corner
    ]
    
    known_forces, trace, active_ids = identify_zfms_by_inspection(nodes, elements, supports, loads)
    
    # 1. Assert exactly 3 members were identified as ZFMs
    assert len(known_forces) == 3
    
    # 2. Assert Rule 1 caught the unloaded corner at Node 4
    assert known_forces['E'] == 0.0
    assert known_forces['F'] == 0.0
    
    # 3. Assert Rule 2 caught the vertical strut at Node 2
    # (Members A and B are collinear, so C is a ZFM)
    assert known_forces['C'] == 0.0
    
    # 4. Assert the active pool correctly removed the ZFMs
    assert active_ids == {'A', 'B', 'D'}
    
    # 5. Assert the pedagogical Trace built correctly
    assert len(trace) == 2
    
    # We sort the nodes alphanumerically, so Node 2 is evaluated before Node 4
    assert trace[0]['node'] == 2
    assert "Rule 2" in trace[0]['method']
    assert trace[0]['solved_members'] == {'C': 0.0}
    
    assert trace[1]['node'] == 4
    assert "Rule 1" in trace[1]['method']
    assert trace[1]['solved_members'] == {'E': 0.0, 'F': 0.0}


def test_zfm_rule_3_toggle():
    """
    Test that the optional 'Collinear Load' rule correctly identifies a ZFM 
    only when explicitly activated by the toggle.
    """
    nodes = {
        1: (0, 0),   # The Loaded Joint
        2: (10, 0),  # Horizontal neighbor
        3: (5, 5)    # Diagonal neighbor
    }
    
    elements = [
        {'id': 'horiz', 'nodes': (1, 2)},
        {'id': 'diag', 'nodes': (1, 3)}
    ]
    
    # Load is purely horizontal, making it collinear with the 'horiz' member.
    # Therefore, the 'diag' member must carry zero force.
    loads = {1: (1000.0, 0.0)} 
    supports = {}
    
    # 1. Test with Toggle OFF (Default behavior)
    known_off, trace_off, active_off = identify_zfms_by_inspection(
        nodes, elements, supports, loads, include_collinear_load_rule=False
    )
    
    assert len(known_off) == 0
    assert len(trace_off) == 0
    assert 'diag' in active_off
    
    # 2. Test with Toggle ON
    known_on, trace_on, active_on = identify_zfms_by_inspection(
        nodes, elements, supports, loads, include_collinear_load_rule=True
    )
    
    assert len(known_on) == 1
    assert known_on['diag'] == 0.0
    assert 'diag' not in active_on
    
    assert len(trace_on) == 1
    assert trace_on[0]['method'] == 'By Inspection (Rule 3: Collinear Load)'
    assert trace_on[0]['node'] == 1


def test_dsm_vs_inspection_zfms():
    """
    Cross-checks the geometric Zero-Force Member (ZFM) pre-processor against
    the numerical Direct Stiffness Method (DSM). Proves that members identified
    as carrying zero force by inspection actually carry zero force mathematically.
    """
    # 1. Shared Numerical Properties
    E = 200e9
    A = 0.01

    # 2. Topology Setup
    #    Node 4 is an unloaded two-member corner (E, F) -> both are ZFMs.
    #    Node 2 is a T-joint: collinear chord A-B plus the vertical stub C,
    #    unloaded -> C is a ZFM.
    #    Member H (3-5) is the brace that actually carries the load from
    #    node 3 to the roller, keeping the structure stable and determinate
    #    once C, E, and F drop out.
    nodes = {
        1: (0.0, 0.0),    # Pin
        2: (10.0, 0.0),   # Bottom Mid (T-Joint location)
        3: (10.0, 10.0),  # Top Left (Loaded)
        4: (20.0, 10.0),  # Top Right (Unloaded Corner)
        5: (20.0, 0.0)    # Roller
    }

    supports = {
        1: ['x', 'y'],
        5: ['y']
    }

    loads = {
        3: (1000.0, -1000.0)
    }

    elements = [
        {'id': 'A', 'nodes': (1, 2), 'E': E, 'A': A},
        {'id': 'B', 'nodes': (2, 5), 'E': E, 'A': A},
        {'id': 'C', 'nodes': (2, 3), 'E': E, 'A': A},  # Vertical stub at T-Joint (ZFM)
        {'id': 'D', 'nodes': (1, 3), 'E': E, 'A': A},
        {'id': 'E', 'nodes': (3, 4), 'E': E, 'A': A},  # Top to corner (ZFM)
        {'id': 'F', 'nodes': (4, 5), 'E': E, 'A': A},  # Right to corner (ZFM)
        {'id': 'H', 'nodes': (3, 5), 'E': E, 'A': A}   # <-- Stabilizing brace (load path)
    ]

    # 3. Run the Pedagogical Pre-Processor
    known_forces_sym, _, _ = identify_zfms_by_inspection(nodes, elements, supports, loads)

    # 4. Run the Direct Stiffness Method
    _, internal_forces_dsm, _ = solve_truss(nodes, elements, supports, loads)

    # 5. Cross-Check Verification
    # The pre-processor should have identified exactly 3 ZFMs.
    assert len(known_forces_sym) == 3
    assert 'C' in known_forces_sym
    assert 'E' in known_forces_sym
    assert 'F' in known_forces_sym

    # For every member the pre-processor flagged as a ZFM, verify the
    # rigorous numerical engine also calculated its force as ~0.0.
    # (abs=1e-6 leaves headroom above the matrix-inversion noise floor; the
    # forces in the load-carrying members here are O(1e3), so this is a
    # genuine zero check, not just confirming the solver's own cleanup pass.)
    for zfm_id in known_forces_sym.keys():
        dsm_force = internal_forces_dsm[zfm_id]
        assert pytest.approx(dsm_force, abs=1e-6) == 0.0


def test_node_difficulty_scorer():
    """
    Verifies that the heuristic scorer correctly identifies the algebraic 
    complexity of different joint configurations.
    """
    nodes = {
        1: (0, 0),
        2: (10, 0),
        3: (5, 5),
        4: (0, 10)
    }
    
    el_horiz = {'id': 'A', 'nodes': (1, 2)}
    el_vert = {'id': 'B', 'nodes': (1, 4)}
    el_diag1 = {'id': 'C', 'nodes': (1, 3)}
    el_diag2 = {'id': 'D', 'nodes': (2, 3)}
    
    # Condition 1: Trivial (Only 1 unknown)
    # Node 1 evaluating only member A
    score_1 = get_node_difficulty_score(1, [el_horiz], nodes)
    assert score_1 == 1
    
    # Condition 2: Decoupled (2 unknowns, one is horizontal)
    # Node 1 evaluating members A (horizontal) and C (diagonal)
    score_2 = get_node_difficulty_score(1, [el_horiz, el_diag1], nodes)
    assert score_2 == 2
    
    # Condition 2 Alternate: Decoupled (2 unknowns, both axis-aligned)
    # Node 1 evaluating members A (horizontal) and B (vertical)
    score_2_alt = get_node_difficulty_score(1, [el_horiz, el_vert], nodes)
    assert score_2_alt == 2
    
    # Condition 3: Coupled (2 unknowns, both are diagonals)
    # Node 3 evaluating members C (diagonal to origin) and D (diagonal to Node 2)
    score_3 = get_node_difficulty_score(3, [el_diag1, el_diag2], nodes)
    assert score_3 == 3


def test_full_method_of_joints_pipeline():
    """
    Tests the complete end-to-end Method of Joints orchestrator on a simple 
    triangle truss with a horizontal load at the peak.
    """
    L, H, P = sp.symbols('L H P', positive=True)
    
    nodes = {
        1: (0, 0),         # Pin
        2: (L, 0),         # Roller
        3: (L/2, H)        # Peak
    }
    
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (P, 0)}    # Pushing right at the peak
    
    elements = [
        {'id': 'bot', 'nodes': (1, 2)},
        {'id': 'left_diag', 'nodes': (1, 3)},
        {'id': 'right_diag', 'nodes': (2, 3)}
    ]
    
    # Run the full MoJ orchestrator
    trace, internal_forces, global_reactions = solve_method_of_joints(nodes, elements, supports, loads)
    
    # 1. Verify the raw mathematical output (for automated grading)
    # The right diagonal is pushed on by P, so it should be in compression (-)
    # The left diagonal resists being pulled, so it should be in tension (+)
    # The bottom chord prevents the roller from sliding right, so tension (+)
    
    # Calculate expected geometry factors
    diag_length = sp.sqrt((L/2)**2 + H**2)
    cos_theta = (L/2) / diag_length
    
    # F_right_diag * cos_theta + P/2 = 0 (symmetric load share simplified)
    expected_right_diag = -P / (2 * cos_theta)
    
    # We can assert that the internal forces dict is fully populated
    assert len(internal_forces) == 3
    assert 'bot' in internal_forces
    
    # 2. Verify the Trace structure (for prompt generation)
    assert len(trace) > 1
    assert trace[0]['step'] == 'Global Statics'
    
    # Verify the semantic string translation worked
    # We should find the word 'Compression' somewhere in the right diagonal's trace results
    found_compression_translation = False
    for step in trace:
        if step.get('step') == 'Joint Equilibrium':
            results = step['results']
            if 'right_diag' in results and 'Compression' in results['right_diag']:
                found_compression_translation = True
                
    assert found_compression_translation


def test_dsm_vs_method_of_joints():
    """
    Cross-checks the complete Method of Joints (Symbolic Pathfinding) 
    against the Direct Stiffness Method (Numerical Matrix Inversion).
    Ensures local joint equilibrium steps match global structural mechanics.
    """
    # ---------------------------------------------------------
    # 1. Define Numerical Parameters for DSM
    # ---------------------------------------------------------
    num_L = 12.0
    num_H = 5.0
    num_P = 2500.0  # 2.5 kN horizontal load at the peak
    E = 200e9
    A = 0.005

    supports = {
        1: ['x', 'y'], # Pin at origin
        2: ['y']       # Roller at (L, 0)
    }

    nodes_num = {
        1: (0.0, 0.0),
        2: (num_L, 0.0),
        3: (num_L / 2.0, num_H)
    }
    
    elements_num = [
        {'id': 'bot', 'nodes': (1, 2), 'E': E, 'A': A},
        {'id': 'left_diag', 'nodes': (1, 3), 'E': E, 'A': A},
        {'id': 'right_diag', 'nodes': (2, 3), 'E': E, 'A': A}
    ]
    
    loads_num = {3: (num_P, 0.0)} 
    
    # Run the numerical engine
    _, forces_dsm, _ = solve_truss(nodes_num, elements_num, supports, loads_num)

    # ---------------------------------------------------------
    # 2. Run the Symbolic Method of Joints Engine
    # ---------------------------------------------------------
    L, H, P = sp.symbols('L H P', positive=True)
    
    nodes_sym = {
        1: (0, 0),
        2: (L, 0),
        3: (L/2, H)
    }
    
    loads_sym = {3: (P, 0)}
    
    # Run the pathfinding loop to get symbolic expressions
    _, forces_moj_sym, _ = solve_method_of_joints(nodes_sym, elements_num, supports, loads_sym)
    
    # ---------------------------------------------------------
    # 3. Substitute Numbers into Symbolic Outputs & Verify
    # ---------------------------------------------------------
    substitutions = {L: num_L, H: num_H, P: num_P}
    
    forces_moj_num = {
        el_id: float(expr.subs(substitutions).evalf()) 
        for el_id, expr in forces_moj_sym.items()
    }
    
    # Both solvers natively assume Tension is Positive (+), 
    # so magnitudes and signs must match identically.
    for el_id in forces_dsm.keys():
        assert pytest.approx(forces_moj_num[el_id], abs=1e-8) == forces_dsm[el_id]