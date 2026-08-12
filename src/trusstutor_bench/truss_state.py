import sympy as sp
from typing import Dict, Any, List, Set, Tuple


class TrussState:
    def __init__(self, 
                 nodes: Dict[Any, Tuple[Any, Any]], 
                 elements: List[Dict[str, Any]], 
                 supports: Dict[Any, List[str]], 
                 loads: Dict[Any, Tuple[Any, Any]]):
        """
        Initializes the state container for a hybrid truss solution strategy.
        Accepts both numeric values and SymPy symbols for coordinates and forces.
        """
        # --- Immutable Problem Topology ---
        # Normalize node coordinates to exact rationals to prevent float drift
        self.nodes = {
            nid: (sp.nsimplify(x, rational=True), sp.nsimplify(y, rational=True)) 
            for nid, (x, y) in nodes.items()
        }        
        self.elements = elements
        self.supports = supports
        self.loads = loads
        
        # --- Dynamic Solved States ---
        self.known_forces: Dict[str, Any] = {}
        self.global_reactions: Dict[str, Any] = {}
        
        # Track which internal elements remain unsolved (ignoring abstract boundaries like 'wall')
        self.active_element_ids: Set[str] = set(
            el['id'] for el in elements if el['id'] != 'wall'
        )
        
        # --- Pedagogical Solution Trace Ledger ---
        self.trace: List[Dict[str, Any]] = []

    def log_step(self, step_type: str, target: Any, results: Dict[str, Any], explanation: str) -> None:
        """
        Appends a standardized pedagogical ledger entry to the trace.
        """
        self.trace.append({
            "step_type": step_type,
            "target": target,
            "results": {k: str(v) for k, v in results.items()},
            "explanation": explanation
        })

    def copy(self) -> 'TrussState':
        """
        Creates a duplicate of the dynamic state. This is highly useful for 
        testing branching solution strategies without polluting the main state.
        """
        new_state = TrussState(self.nodes, self.elements, self.supports, self.loads)
        new_state.known_forces = self.known_forces.copy()
        new_state.global_reactions = self.global_reactions.copy()
        new_state.active_element_ids = self.active_element_ids.copy()
        
        # Deep copy the trace list to prevent shared references
        new_state.trace = [entry.copy() for entry in self.trace]
        
        return new_state

    def get_final_answers(self) -> Dict[str, Any]:
        """
        Returns the compiled ground-truth values for automated grading evaluation.
        """
        return {
            "internal_forces": self.known_forces,
            "global_reactions": self.global_reactions
        }


