import sympy as sp
import pytest
from trusstutor_bench.math_helpers import get_unit_vector, are_collinear, get_node_difficulty_score


def test_get_unit_vector():
    """Verifies pointing-away vectors for both numeric and symbolic coordinates."""
    nodes_num = {1: (0, 0), 2: (3, 4)}
    el = {'nodes': (1, 2)}
    
    # Test numeric evaluation
    # (Cast to float to safely compare SymPy exact Rationals to Python floats)
    c, s = get_unit_vector(1, el, nodes_num)
    assert float(c) == 3/5
    assert float(s) == 4/5
    
    # Check that it reverses correctly when pointing from the other node
    c_rev, s_rev = get_unit_vector(2, el, nodes_num)
    assert float(c_rev) == -3/5
    assert float(s_rev) == -4/5
    
    # Test symbolic evaluation
    L, H = sp.symbols('L H', positive=True)
    nodes_sym = {1: (0, 0), 2: (L, H)}
    c_sym, s_sym = get_unit_vector(1, el, nodes_sym)
    
    expected_length = sp.sqrt(L**2 + H**2)
    assert c_sym == L / expected_length
    assert s_sym == H / expected_length


def test_are_collinear():
    """Verifies that the cross-product logic correctly identifies lines of action."""
    nodes = {
        1: (0, 0), 
        2: (10, 0),   # Horizontal right
        3: (-5, 0),   # Horizontal left
        4: (5, 5)     # Diagonal
    }
    
    el_right = {'nodes': (1, 2)}
    el_left = {'nodes': (1, 3)}
    el_diag = {'nodes': (1, 4)}
    
    assert are_collinear(1, el_right, el_left, nodes) is True
    
    assert are_collinear(1, el_right, el_diag, nodes) is False


def test_node_difficulty_scorer():
    """Verifies algebraic difficulty categorization based on direction cosines."""
    nodes = {
        1: (0, 0),
        2: (10, 0),
        3: (5, 5),
        4: (-5, 5)
    }
    
    el_horiz = {'id': 'A', 'nodes': (1, 2)}
    el_diag1 = {'id': 'B', 'nodes': (1, 3)}
    el_diag2 = {'id': 'C', 'nodes': (1, 4)}
    
    # Tier 1: Single Unknown
    assert get_node_difficulty_score(1, [el_horiz], nodes) == 1
    
    # Tier 2: Two Unknowns, one is purely horizontal (decoupled)
    assert get_node_difficulty_score(1, [el_horiz, el_diag1], nodes) == 2
    
    # Tier 3: Two Unknowns, both are diagonals (coupled)
    assert get_node_difficulty_score(1, [el_diag1, el_diag2], nodes) == 3
    
    # Error Case: Too many unknowns
    with pytest.raises(ValueError):
        get_node_difficulty_score(1, [el_horiz, el_diag1, el_diag2], nodes)
