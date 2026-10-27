import sympy as sp
from typing import Dict, Any, List, Set, Tuple


class TrussState:
    """
    Holds a truss problem and its solution progress: forces, reactions and the step trace.
    """

    def __init__(self,
                 nodes: Dict[Any, Tuple[Any, Any]],
                 elements: List[Dict[str, Any]],
                 supports: Dict[Any, List[str]],
                 loads: Dict[Any, Tuple[Any, Any]]):
        """
        Stores the problem and starts with nothing solved.

        Args:
            nodes (dict): Node ID -> (x, y), numeric or SymPy values.
            elements (list): Element dictionaries with "id" and "nodes".
            supports (dict): Node ID -> list of constrained directions.
            loads (dict): Node ID -> (Fx, Fy).
        """
        # Exact rationals avoid float drift.
        self.nodes = {
            nid: (sp.nsimplify(x, rational=True), sp.nsimplify(y, rational=True))
            for nid, (x, y) in nodes.items()
        }
        self.elements = elements
        self.supports = supports
        self.loads = loads

        self.known_forces: Dict[str, Any] = {}
        self.global_reactions: Dict[str, Any] = {}

        # 'wall' is a boundary, not a member.
        self.active_element_ids: Set[str] = set(
            el['id'] for el in elements if el['id'] != 'wall'
        )

        self.trace: List[Dict[str, Any]] = []

    def log_step(self, step_type: str, target: Any, results: Dict[str, Any], explanation: str) -> None:
        """
        Appends one entry to the solution trace.

        Args:
            step_type (str): Kind of step.
            target (Any): Node or members the step solves.
            results (dict): Solved values, stored as strings.
            explanation (str): Short description of the step.
        """
        self.trace.append({
            "step_type": step_type,
            "target": target,
            "results": {k: str(v) for k, v in results.items()},
            "explanation": explanation
        })

    def copy(self) -> 'TrussState':
        """
        Returns a copy of the state. The trace is copied, so the two states do not share it.

        Returns:
            TrussState: The copy.
        """
        new_state = TrussState(self.nodes, self.elements, self.supports, self.loads)
        new_state.known_forces = self.known_forces.copy()
        new_state.global_reactions = self.global_reactions.copy()
        new_state.active_element_ids = self.active_element_ids.copy()

        new_state.trace = [entry.copy() for entry in self.trace]

        return new_state

    def get_final_answers(self) -> Dict[str, Any]:
        """
        Returns the solved values used for grading.

        Returns:
            dict: "internal_forces" and "global_reactions".
        """
        return {
            "internal_forces": self.known_forces,
            "global_reactions": self.global_reactions
        }
