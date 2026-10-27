import pytest
import sympy as sp
from trusstutor_bench.truss_state import TrussState


@pytest.fixture
def sample_topology():
    """Provides a basic truss topology for state testing."""
    nodes = {1: (0, 0), 2: (10, 0), 3: (5, 5)}
    elements = [
        {'id': 'wall', 'nodes': (1, 1)}, 
        {'id': 'bot', 'nodes': (1, 2)},
        {'id': 'diag1', 'nodes': (1, 3)},
        {'id': 'diag2', 'nodes': (2, 3)}
    ]
    supports = {1: ['x', 'y'], 2: ['y']}
    loads = {3: (0, -1000)}
    return nodes, elements, supports, loads


def test_state_initialization(sample_topology):
    """Verifies that the state initializes and correctly filters active elements."""
    nodes, elements, supports, loads = sample_topology
    state = TrussState(nodes, elements, supports, loads)
    
    assert state.nodes == nodes
    assert state.loads == loads
    
    assert 'wall' not in state.active_element_ids
    assert state.active_element_ids == {'bot', 'diag1', 'diag2'}
    
    assert len(state.known_forces) == 0
    assert len(state.global_reactions) == 0
    assert len(state.trace) == 0


def test_state_logging(sample_topology):
    """Verifies that pedagogical steps are correctly formatted and appended."""
    nodes, elements, supports, loads = sample_topology
    state = TrussState(nodes, elements, supports, loads)
    
    state.log_step(
        step_type="Joint Equilibrium",
        target=3,
        results={'diag1': -500, 'diag2': -500},
        explanation="Solved the peak joint."
    )
    
    assert len(state.trace) == 1
    assert state.trace[0]['step_type'] == "Joint Equilibrium"
    assert state.trace[0]['target'] == 3
    assert state.trace[0]['results']['diag1'] == "-500"


def test_state_copying(sample_topology):
    """Verifies that copied states do not share mutable references."""
    nodes, elements, supports, loads = sample_topology
    state_A = TrussState(nodes, elements, supports, loads)
    
    state_B = state_A.copy()
    
    state_A.known_forces['bot'] = 1000
    state_A.active_element_ids.remove('bot')
    state_A.log_step("Test", 1, {}, "Mutated State A")
    
    assert 'bot' not in state_B.known_forces
    assert 'bot' in state_B.active_element_ids
    assert len(state_B.trace) == 0


def test_get_final_answers(sample_topology):
    """Verifies that the state correctly packages the final answer dictionaries."""
    nodes, elements, supports, loads = sample_topology
    state = TrussState(nodes, elements, supports, loads)
    
    # Manually inject some solved data
    state.known_forces['bot'] = 500.0
    state.global_reactions['R_1_x'] = -500.0
    
    answers = state.get_final_answers()
    
    assert 'internal_forces' in answers
    assert 'global_reactions' in answers
    assert answers['internal_forces']['bot'] == 500.0
    assert answers['global_reactions']['R_1_x'] == -500.0


def test_logging_sympy_types(sample_topology):
    """Verifies that abstract SymPy expressions are correctly cast to strings in the trace."""
    nodes, elements, supports, loads = sample_topology
    state = TrussState(nodes, elements, supports, loads)
    
    P, L = sp.symbols('P L')
    symbolic_force = P * L / 2
    
    # Log a step containing a pure SymPy expression
    state.log_step(
        step_type="Symbolic Evaluation",
        target=1,
        results={'bot': symbolic_force},
        explanation="Testing sympy logging."
    )
    
    # The result should be safely cast to a string format like "L*P/2"
    assert isinstance(state.trace[0]['results']['bot'], str)
    assert "P" in state.trace[0]['results']['bot']
