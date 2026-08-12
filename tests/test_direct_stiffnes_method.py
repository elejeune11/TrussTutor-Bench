import numpy as np
import pytest
from trusstutor_bench.direct_stiffness_method import main
from trusstutor_bench.direct_stiffness_method import compute_element_properties
from trusstutor_bench.direct_stiffness_method import compute_element_stiffness
from trusstutor_bench.direct_stiffness_method import assemble_global_stiffness
from trusstutor_bench.direct_stiffness_method import apply_boundary_conditions
from trusstutor_bench.direct_stiffness_method import compute_internal_forces
from trusstutor_bench.direct_stiffness_method import assemble_force_vector
from trusstutor_bench.direct_stiffness_method import solve_truss
from trusstutor_bench.direct_stiffness_method import check_stability_and_determinacy


def test_main_output(capsys):
    """
    Test that main() prints the correct initialization string to stdout.
    """
    # 1. Execute the function
    main()
    # 2. Read the captured output
    captured = capsys.readouterr()
    # 3. Assert the output matches exactly (note the implicit newline from print)
    assert captured.out == "Hello from trusstutor-support-code!\n"


def test_compute_element_properties_horizontal():
    """Test a purely horizontal member pointing in the positive X direction."""
    node_i = (0.0, 0.0)
    node_j = (5.0, 0.0)
    
    # Expected: Length = 5.0, cos(0) = 1.0, sin(0) = 0.0
    L, c, s = compute_element_properties(node_i, node_j)
    
    assert pytest.approx(L) == 5.0
    assert pytest.approx(c) == 1.0
    assert pytest.approx(s) == 0.0


def test_compute_element_properties_vertical_negative():
    """Test a purely vertical member pointing in the negative Y direction."""
    node_i = (2.0, 3.0)
    node_j = (2.0, 0.0)
    
    # Expected: Length = 3.0, cos(270 deg) = 0.0, sin(270 deg) = -1.0
    L, c, s = compute_element_properties(node_i, node_j)
    
    assert pytest.approx(L) == 3.0
    assert pytest.approx(c) == 0.0
    assert pytest.approx(s) == -1.0


def test_compute_element_properties_3_4_5_triangle():
    """Test a standard 3-4-5 triangle quadrant orientation."""
    node_i = (1.0, 1.0)
    node_j = (4.0, 5.0)  # delta_x = 3, delta_y = 4
    
    # Expected: Length = 5.0, cos = 3/5 = 0.6, sin = 4/5 = 0.8
    L, c, s = compute_element_properties(node_i, node_j)
    
    assert pytest.approx(L) == 5.0
    assert pytest.approx(c) == 0.6
    assert pytest.approx(s) == 0.8


def test_element_stiffness_horizontal():
    """
    A horizontal member (c=1, s=0) should only have stiffness in the x-degrees of freedom.
    The y-degrees of freedom (rows/cols 1 and 3) must be entirely zeros.
    """
    E = 200e9  # Steel (Pa)
    A = 0.01   # Area (m^2)
    L = 5.0    # Length (m)
    c, s = 1.0, 0.0 
    
    k = (E * A) / L
    
    K_e = compute_element_stiffness(E, A, L, c, s)
    
    # Expected analytical matrix
    expected = k * np.array([
        [ 1,  0, -1,  0],
        [ 0,  0,  0,  0],
        [-1,  0,  1,  0],
        [ 0,  0,  0,  0]
    ])
    
    assert np.allclose(K_e, expected)


def test_element_stiffness_vertical():
    """
    A vertical member (c=0, s=1) should only have stiffness in the y-degrees of freedom.
    The x-degrees of freedom (rows/cols 0 and 2) must be entirely zeros.
    """
    E = 200e9 
    A = 0.01  
    L = 4.0   
    c, s = 0.0, 1.0 
    
    k = (E * A) / L
    
    K_e = compute_element_stiffness(E, A, L, c, s)
    
    # Expected analytical matrix
    expected = k * np.array([
        [ 0,  0,  0,  0],
        [ 0,  1,  0, -1],
        [ 0,  0,  0,  0],
        [ 0, -1,  0,  1]
    ])
    
    assert np.allclose(K_e, expected)


def test_element_stiffness_diagonal():
    """
    A diagonal member at 45 degrees (c = 1/sqrt(2), s = 1/sqrt(2)).
    The squared and cross terms (c^2, s^2, cs) will all equal exactly 0.5.
    This ensures all off-diagonal signs are correct.
    """
    E = 200e9 
    A = 0.01  
    L = np.sqrt(8) # e.g., a 2x2 grid diagonal
    
    # 45 degrees
    c = 1.0 / np.sqrt(2)
    s = 1.0 / np.sqrt(2)
    
    k = (E * A) / L
    
    K_e = compute_element_stiffness(E, A, L, c, s)
    
    # Expected analytical matrix (every term is +/- 0.5)
    expected = k * np.array([
        [ 0.5,  0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5,  0.5],
        [-0.5, -0.5,  0.5,  0.5]
    ])
    
    assert np.allclose(K_e, expected)


def test_global_stiffness_singularity():
    """
    An unsupported truss has rigid body motion. 
    Therefore, its global stiffness matrix must be singular 
    (determinant is effectively zero, matrix rank is less than 2N).
    """
    nodes = {
        1: (0.0, 0.0),
        2: (10.0, 0.0),
        3: (5.0, 8.66)
    }
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 200e9, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 200e9, 'A': 0.01},
        {'id': 3, 'nodes': (1, 3), 'E': 200e9, 'A': 0.01}
    ]
    
    K = assemble_global_stiffness(nodes, elements)
    
    # Check if the matrix is singular by computing its rank.
    # A fully constrained 2D truss with N nodes has 2N DOFs. 
    # Unconstrained, it has 3 rigid body modes, so rank should be (2N - 3).
    dofs = 2 * len(nodes)
    rank = np.linalg.matrix_rank(K)
    
    assert rank == (dofs - 3), f"Expected rank {dofs - 3}, but got {rank}. Matrix is not properly singular."


def test_apply_boundary_conditions_size():
    """
    Tests that a 3-node system (6 total DOFs) with a pin (2 constraints)
    and a roller (1 constraint) correctly reduces to 3 free DOFs.
    """
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 5)}
    elements = [{'id': 1, 'nodes': (1, 2), 'E': 200e9, 'A': 0.01}]
    
    K_global = assemble_global_stiffness(nodes, elements)
    F_global = np.zeros(2 * len(nodes))
    
    # Pin at Node 1 (fixes x, y), Roller at Node 2 (fixes y) -> 3 constraints total
    supports = {
        1: ['x', 'y'],
        2: ['y']
    }
    
    K_reduced, F_reduced, free_dofs = apply_boundary_conditions(nodes, K_global, F_global, supports)
    
    # 6 total DOFs - 3 constrained DOFs = 3 free DOFs
    assert K_reduced.shape == (3, 3)
    assert len(F_reduced) == 3
    assert len(free_dofs) == 3


def test_reduced_matrix_invertibility():
    """
    Verifies that applying proper boundary conditions removes singularity,
    meaning the determinant is non-zero and the matrix can be inverted.
    """
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 5)}
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 2e11, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 2e11, 'A': 0.01},
        {'id': 3, 'nodes': (1, 3), 'E': 2e11, 'A': 0.01}
    ]
    
    K_global = assemble_global_stiffness(nodes, elements)
    F_global = np.zeros(2 * len(nodes))
    
    supports = {
        1: ['x', 'y'],
        2: ['y']
    }
    
    K_reduced, _, _ = apply_boundary_conditions(nodes, K_global, F_global, supports)
    
    # Check condition number or determinant to ensure invertibility
    det = np.linalg.det(K_reduced)
    assert abs(det) > 1e-3, f"Reduced matrix is still singular! Det: {det}"


def test_internal_forces_tension():
    """
    A single horizontal member pulled by a displacement.
    If Node 2 moves 0.001m to the right, the member is in tension.
    """
    nodes = {1: (0, 0), 2: (5, 0)}
    elements = [{'id': 1, 'nodes': (1, 2), 'E': 200e9, 'A': 0.01}]
    
    # Node 1 is fixed (0,0). Node 2 moves 0.001m in X, 0 in Y.
    U_global = np.array([0.0, 0.0, 0.001, 0.0])
    
    forces = compute_internal_forces(nodes, elements, U_global)
    
    # Expected Force = (E*A/L) * delta_L
    # = (200e9 * 0.01 / 5) * 0.001 = 400,000 N (400 kN)
    assert pytest.approx(forces[1]) == 400000.0
    assert forces[1] > 0  # Must be positive for tension


def test_internal_forces_zero_force_member():
    """
    Pedagogical Edge Case: A vertical member connecting to a straight horizontal chord
    with no external vertical load. The vertical member MUST be a Zero-Force Member.
    """
    nodes = {
        1: (0, 0),    # Left support
        2: (10, 0),   # Right support
        3: (5, 0),    # Midpoint
        4: (5, 5)     # Top node
    }
    elements = [
        {'id': 1, 'nodes': (1, 3), 'E': 200e9, 'A': 0.01}, # Left chord
        {'id': 2, 'nodes': (3, 2), 'E': 200e9, 'A': 0.01}, # Right chord
        {'id': 3, 'nodes': (3, 4), 'E': 200e9, 'A': 0.01}  # Vertical ZFM
    ]
    
    # Simulate a horizontal load stretching the bottom chord.
    # Node 1 is fixed. Node 2 and 3 move right. Node 4 rides along horizontally.
    # Critically, there are NO vertical displacements (Y=0 for all nodes).
    U_global = np.array([
        0.0, 0.0,      # Node 1
        0.002, 0.0,    # Node 2
        0.001, 0.0,    # Node 3
        0.001, 0.0     # Node 4 (moves with Node 3, so no vertical stretch)
    ])
    
    forces = compute_internal_forces(nodes, elements, U_global)
    
    # The vertical member (Element 3) should have exactly 0.0 force
    assert forces[3] == 0.0


def test_assemble_force_vector():
    """
    Verifies that external load tuples (Fx, Fy) are correctly mapped 
    to the 2N global force vector based on arbitrary Node IDs.
    """
    nodes = {
        1: (0.0, 0.0),    # Index 0: DOFs 0, 1
        2: (5.0, 0.0),    # Index 1: DOFs 2, 3
        3: (2.5, 4.0)     # Index 2: DOFs 4, 5
    }
    
    # Apply a 100 N horizontal load to Node 2, and a 500 N downward load to Node 3.
    # Node 1 has no loads applied.
    loads = {
        2: (100.0, 0.0),
        3: (0.0, -500.0)
    }
    
    F_global = assemble_force_vector(nodes, loads)
    
    # The expected vector should be length 6 (2 * 3 nodes)
    assert len(F_global) == 6
    
    # Expected analytical force vector
    expected_F = np.array([
        0.0,    0.0,      # Node 1 (Fx, Fy)
        100.0,  0.0,      # Node 2 (Fx, Fy)
        0.0,   -500.0     # Node 3 (Fx, Fy)
    ])
    
    assert np.allclose(F_global, expected_F)


def test_solve_truss_equilibrium():
    """
    End-to-End Test: A 3-bar triangular truss subjected to a horizontal load.
    Verifies that the sum of the reactions exactly equals the applied loads
    (Sum of Fx = 0, Sum of Fy = 0).
    """
    # 3-4-5 Triangle geometry
    nodes = {
        1: (0.0, 0.0),    # Bottom left (Pinned)
        2: (3.0, 0.0),    # Bottom right (Roller)
        3: (0.0, 4.0)     # Top left (Loaded)
    }
    
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 200e9, 'A': 0.01}, # Bottom chord
        {'id': 2, 'nodes': (2, 3), 'E': 200e9, 'A': 0.01}, # Hypotenuse
        {'id': 3, 'nodes': (1, 3), 'E': 200e9, 'A': 0.01}  # Vertical chord
    ]
    
    supports = {
        1: ['x', 'y'], # Pin
        2: ['y']       # Roller
    }
    
    # Apply 1000 N to the right at the top node
    loads = {
        3: (1000.0, 0.0)
    }
    
    U_global, internal_forces, reactions = solve_truss(nodes, elements, supports, loads)
    
    # Verify Global Equilibrium
    # Sum of all applied loads
    total_applied_fx = sum(fx for fx, fy in loads.values())
    total_applied_fy = sum(fy for fx, fy in loads.values())
    
    # Sum of all calculated reactions (Indices: 0,2,4 are X; 1,3,5 are Y)
    total_reaction_fx = np.sum(reactions[0::2])
    total_reaction_fy = np.sum(reactions[1::2])
    
    # Equilibrium: Applied + Reactions must equal 0
    assert pytest.approx(total_applied_fx + total_reaction_fx) == 0.0
    assert pytest.approx(total_applied_fy + total_reaction_fy) == 0.0
    
    # Verify specific boundary condition behaviors:
    # Node 2 is a roller (fixes Y, free X). Therefore, Reaction X at Node 2 MUST be 0.
    # Node 2 X-DOF index is 2 (3rd element in array)
    assert reactions[2] == 0.0


def test_check_indeterminate():
    """Test a truss with cross-bracing that makes it statically indeterminate."""
    nodes = {1: (0, 0), 2: (5, 0), 3: (5, 5), 4: (0, 5)}
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 2e11, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 2e11, 'A': 0.01},
        {'id': 3, 'nodes': (3, 4), 'E': 2e11, 'A': 0.01},
        {'id': 4, 'nodes': (4, 1), 'E': 2e11, 'A': 0.01},
        {'id': 5, 'nodes': (1, 3), 'E': 2e11, 'A': 0.01},
        {'id': 6, 'nodes': (2, 4), 'E': 2e11, 'A': 0.01} # Redundant diagonal
    ]
    supports = {1: ['x', 'y'], 2: ['y']} # 3 reaction forces
    
    # j=4, m=6, r=3. 2j (8) < m+r (9). Indeterminate to the 1st degree.
    diagnostic = check_stability_and_determinacy(nodes, elements, supports)
    
    assert diagnostic["solvable_by_hand"] is False
    assert diagnostic["is_singular"] is False
    assert "Degree 1" in diagnostic["status"]


def test_check_deficient_unstable():
    """Test a truss that lacks enough members to form stable triangles."""
    nodes = {1: (0, 0), 2: (5, 0), 3: (5, 5), 4: (0, 5)}
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 2e11, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 2e11, 'A': 0.01},
        {'id': 3, 'nodes': (3, 4), 'E': 2e11, 'A': 0.01},
        {'id': 4, 'nodes': (4, 1), 'E': 2e11, 'A': 0.01}
        # Missing any diagonal bracing. It's a collapsing square.
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    
    diagnostic = check_stability_and_determinacy(nodes, elements, supports)
    
    assert diagnostic["solvable_by_hand"] is False
    assert "Unstable" in diagnostic["status"]


def test_check_geometric_mechanism():
    """
    Test a truss that satisfies Maxwell's Rule (2j = m + r) but is structurally
    unstable due to collinear nodes.
    """
    nodes = {
        1: (0, 0),
        2: (5, 0),
        3: (10, 0) # Collinear with 1 and 2
    }
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 2e11, 'A': 0.01},
        {'id': 2, 'nodes': (2, 3), 'E': 2e11, 'A': 0.01}
    ]
    supports = {1: ['x', 'y'], 3: ['x', 'y']} 
    
    # j=3, m=2, r=4. 2j(6) == m+r(6). Should pass counting, but fail matrix.
    diagnostic = check_stability_and_determinacy(nodes, elements, supports)
    
    assert diagnostic["is_singular"] is True
    assert diagnostic["solvable_by_hand"] is False
    assert "Mechanism" in diagnostic["status"]


def test_solve_truss_more_examples():
    # set up a little triangle
    Fcx = 10
    Fcy = 5
    # geometry
    nodes = {
        1: (1.0, 2.0),
        2: (4.0, 2.0),
        3: (1.5, 3.5)
    }
    elements = [
        {'id': 1, 'nodes': (1, 2), 'E': 2e5, 'A': 0.1},
        {'id': 2, 'nodes': (2, 3), 'E': 2e5, 'A': 0.1},
        {'id': 3, 'nodes': (3, 1), 'E': 2e5, 'A': 0.1},
    ]
    supports = {
        1: ['x', 'y'], 2: ['y']
    }
    loads = {
        3: (Fcx, Fcy)
    }
    # solve
    U_global, internal_forces, reactions = solve_truss(nodes, elements, supports, loads)

    # manual solution
    Ax = -1.0 * Fcx
    By = (1.5 * Fcx - 0.5 * Fcy) / 3.0
    Ay = -1.0 * Fcy - 1.0 * By
    theta = np.atan(1.5/0.5)
    Fca = -1.0 * Ay / np.sin(theta)
    Fab = -1.0 * Ax - 1.0 * Fca * np.cos(theta)
    phi = np.atan(1.5 / 2.5)
    Fbc = (-1.0 * By) / np.sin(phi)

    # compare solution
    Ax_computed = reactions[0]
    Ay_computed = reactions[1]
    By_computed = reactions[3]
    Fab_computed = internal_forces[1]
    Fbc_computed = internal_forces[2]
    Fca_computed = internal_forces[3]

    assert np.isclose(Fab, Fab_computed)
    assert np.isclose(Fbc, Fbc_computed)
    assert np.isclose(Fca, Fca_computed)
    assert np.isclose(Ax, Ax_computed)
    assert np.isclose(Ay, Ay_computed)
    assert np.isclose(By, By_computed)
