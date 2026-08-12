from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class BenchmarkTask:
    """
    A pure, immutable data transfer object (DTO) representing a single Truss problem template.
    Contains zero execution logic.
    """
    benchmark_id: str
    metadata: Dict[str, Any]
    symbolic_definitions: List[Any]
    nodes: Dict[str, tuple]
    elements: List[Dict[str, Any]]
    loads: Dict[str, tuple]
    supports: Dict[str, List[str]]
    problem_schema: Dict[str, Any]
    solution_recipe: Dict[int, Any]
    drawing_variables: Dict[str, Any]

    @classmethod
    def from_module(cls, template_module: Any) -> 'BenchmarkTask':
        """
        Factory method to safely extract attributes from an imported Python module.
        Uses getattr with defaults for safety on optional physical parameters.
        """
        return cls(
            benchmark_id=template_module.METADATA.get("benchmark_id", "UNKNOWN_ID"),
            metadata=template_module.METADATA,
            symbolic_definitions=getattr(template_module, "SYMBOLIC_DEFINITIONS", []),
            nodes=template_module.NODES,
            elements=template_module.ELEMENTS,
            loads=getattr(template_module, "LOADS", {}),
            supports=getattr(template_module, "SUPPORTS", {}),
            problem_schema=template_module.PROBLEM_SCHEMA,
            solution_recipe=template_module.SOLUTION_RECIPE,
            drawing_variables=getattr(template_module, "DRAWING_VARIABLES", {})
        )