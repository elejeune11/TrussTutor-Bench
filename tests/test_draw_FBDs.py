import os
from pathlib import Path
import matplotlib.pyplot as plt

# Import the new drawing primitive
from trusstutor_bench.draw_FBDs import draw_fbd_node
from trusstutor_bench.draw_FBDs import draw_force_vector
from trusstutor_bench.draw_FBDs import draw_cut_stub
from trusstutor_bench.draw_FBDs import draw_intact_element
from trusstutor_bench.draw_FBDs import aggregate_node_forces
from trusstutor_bench.draw_FBDs import generate_section_fbd
from trusstutor_bench.draw_FBDs import generate_global_fbd
from trusstutor_bench.draw_FBDs import generate_node_fbd

import importlib.util
import pytest


def test_draw_fbd_node():
    """Draws a joint with its label and saves a PNG in test_graphics/ for visual inspection."""
    current_dir = Path(__file__).parent
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    output_path = graphics_dir / "test_fbd_node.png"
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal', adjustable='box')
    
    # Optional: Keep the axis on for the test so you can verify 
    # the coordinate placement visually, but remove the bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    draw_fbd_node(ax, node_id="A", x=0.0, y=0.0)
    
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    assert output_path.exists(), f"Image was not saved to {output_path}"
    assert output_path.stat().st_size > 0, "Image file is empty"
    
    print(f"\nVisual inspection ready: {output_path}")


def test_draw_force_vector():
    """Draws force vectors in several orientations and saves a PNG for visual inspection."""

    current_dir = Path(__file__).parent
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    output_path = graphics_dir / "test_force_vector.png"
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off') # Clean canvas for FBD
    
    draw_fbd_node(ax, node_id="A", x=0.0, y=0.0)

    # Quadrant 1: Pointing Up-Right (Tension)
    draw_force_vector(ax, tail_x=0.0, tail_y=0.0, dx=1.5, dy=1.5, label="$F_{AB}$", color="blue")
    
    # Quadrant 2: Pointing Up-Left (Tension)
    draw_force_vector(ax, tail_x=0.0, tail_y=0.0, dx=-2.0, dy=1.0, label="$F_{AC}$", color="green")
    
    # Quadrant 3: Pushing Down-Right into the node (Compression / External Load)
    # Starts at top-left and points to the origin
    draw_force_vector(ax, tail_x=-1.5, tail_y=2.0, dx=1.3, dy=-1.8, label="50 kN", color="red", label_loc="tail")    
    # Quadrant 4: Pointing straight down (Reaction Force)
    draw_force_vector(ax, tail_x=0.0, tail_y=0.0, dx=0.0, dy=-2.0, label="$R_{Ay}$", color="purple")
    
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    assert output_path.exists(), f"Image was not saved to {output_path}"
    assert output_path.stat().st_size > 0, "Image file is empty"
    
    print(f"\nVisual inspection ready: {output_path}")


def test_draw_cut_stub():
    """Draws cut stubs in several directions and saves a PNG for visual inspection."""
    import math
    from pathlib import Path
    import matplotlib.pyplot as plt

    current_dir = Path(__file__).parent
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    output_path = graphics_dir / "test_cut_stub.png"
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    from trusstutor_bench.draw_FBDs import draw_fbd_node
    draw_fbd_node(ax, node_id="B", x=0.0, y=0.0)

    # Stub 1: Pointing directly right (0 degrees)
    draw_cut_stub(ax, start_x=0.0, start_y=0.0, unit_dx=1.0, unit_dy=0.0, stub_length=1.5)
    
    # Stub 2: Pointing straight up (90 degrees)
    draw_cut_stub(ax, start_x=0.0, start_y=0.0, unit_dx=0.0, unit_dy=1.0, stub_length=1.0)
    
    # Stub 3: Pointing diagonally down-right (-45 degrees)
    inv_sqrt2 = 1.0 / math.sqrt(2)
    draw_cut_stub(ax, start_x=0.0, start_y=0.0, unit_dx=inv_sqrt2, unit_dy=-inv_sqrt2, stub_length=1.2)
    
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    assert output_path.exists(), f"Image was not saved to {output_path}"
    assert output_path.stat().st_size > 0, "Image file is empty"
    
    print(f"\nVisual inspection ready: {output_path}")


def test_draw_intact_element():
    """Draws an intact member and saves a PNG for visual inspection."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    current_dir = Path(__file__).parent
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    output_path = graphics_dir / "test_intact_element.png"
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    from trusstutor_bench.draw_FBDs import draw_fbd_node
    draw_fbd_node(ax, node_id="A", x=-1.0, y=0.0)
    draw_fbd_node(ax, node_id="B", x=2.0, y=1.5)
    
    draw_intact_element(ax, x1=-1.0, y1=0.0, x2=2.0, y2=1.5)
    
    ax.set_xlim(-2.0, 3.0)
    ax.set_ylim(-1.0, 2.5)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    assert output_path.exists(), f"Image was not saved to {output_path}"
    assert output_path.stat().st_size > 0, "Image file is empty"
    
    print(f"\nVisual inspection ready: {output_path}")


class MockTrussState:
    """Minimal truss state with loads, reactions and forces, for the aggregator tests."""
    def __init__(self):
        self.nodes = {"A": (0, 0), "B": (10, 0), "C": (5, 5)}
        self.loads = {"A": (50.0, -100.0)}
        self.global_reactions = {"R_A_x": 0, "R_A_y": 50, "R_B_y": 50}
        self.known_forces = {"AB": 25.0}
        
        self.elements = [
            {"id": "AB", "nodes": ("A", "B")},
            {"id": "AC", "nodes": ("A", "C")},
            {"id": "BC", "nodes": ("B", "C")}
        ]


def test_aggregate_node_forces():
    """Collects the loads, reactions and member forces acting on a node."""
    mock_state = MockTrussState()
    
    # Run the aggregator for Node A
    result = aggregate_node_forces(mock_state, node_id="A")
    
    assert len(result["external_loads"]) == 1
    assert result["external_loads"][0]["fx"] == 50.0
    assert result["external_loads"][0]["fy"] == -100.0
    
    assert len(result["support_reactions"]) == 2
    
    # Check that both X and Y reactions were found
    reaction_ids = [r["id"] for r in result["support_reactions"]]
    assert "R_A_x" in reaction_ids
    assert "R_A_y" in reaction_ids
    
    # Verify exact magnitudes
    for r in result["support_reactions"]:
        if r["id"] == "R_A_y":
            assert r["magnitude"] == 50.0
            assert r["axis"] == "y"
            
    assert len(result["internal_forces"]) == 2
    
    # Element AB (Known)
    ab_data = next(item for item in result["internal_forces"] if item["element_id"] == "AB")
    assert ab_data["connected_node"] == "B"
    assert ab_data["status"] == "known"
    assert ab_data["magnitude"] == 25.0
    
    # Element AC (Unknown)
    ac_data = next(item for item in result["internal_forces"] if item["element_id"] == "AC")
    assert ac_data["connected_node"] == "C"
    assert ac_data["status"] == "unknown"
    assert ac_data["magnitude"] is None


def test_aggregate_node_forces_real_truss_0001():
    """Collects the forces on a node of the solved TRUSS_0001 state."""
    current_dir = Path(__file__).parent
    example_path = current_dir.parent / "tasks" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path}")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    recipe = truss_module.SOLUTION_RECIPE
    sampled_vars = {"P": 50.0} 
    
    # Step 1: Initialize the state
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    # Step 2: Solve global reactions
    recipe[2]["function"](state, **recipe[2]["inputs"])
    
    # Step 3: Solve method of joints (this fully populates known_forces)
    recipe[3]["function"](state, **recipe[3]["inputs"])
    
    result_a = aggregate_node_forces(state, node_id="A")

    # In TRUSS_0001, Node A is the bottom-left pin support.
    # It should have both X and Y reactions.
    assert len(result_a["support_reactions"]) == 2
    reaction_ids = [r["id"] for r in result_a["support_reactions"]]
    assert "R_A_x" in reaction_ids
    assert "R_A_y" in reaction_ids
    
    # Verify the specific reaction magnitudes
    for r in result_a["support_reactions"]:
        if r["id"] == "R_A_y":
            assert r["magnitude"] == 25.0
        elif r["id"] == "R_A_x":
            assert r["magnitude"] == 0.0
            
    # Node A connects to members AB and CA.
    assert len(result_a["internal_forces"]) == 2
    internal_ids = [f["element_id"] for f in result_a["internal_forces"]]
    assert "AB" in internal_ids
    assert "CA" in internal_ids
    
    # Because Method of Joints was completed, both should be marked "known"
    # and have a numerical magnitude attached.
    for force in result_a["internal_forces"]:
        assert force["status"] == "known"
        assert force["magnitude"] is not None


def test_generate_global_fbd_examples():
    """Draws the global FBD for TRUSS_0001 to TRUSS_0005."""
    
    current_dir = Path(__file__).parent
    examples_dir = current_dir.parent / "tasks"
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    generated_count = 0
    
    for i in range(1, 6):
        example_name = f"example_TRUSS_{i:04d}.py"
        example_path = examples_dir / example_name
        
        if not example_path.exists():
            continue 
            
        spec = importlib.util.spec_from_file_location(f"example_TRUSS_{i:04d}", example_path)
        truss_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(truss_module)
        
        recipe = truss_module.SOLUTION_RECIPE
        subs_dict = truss_module.DRAWING_VARIABLES
        
        init_step = recipe[1]
        state = init_step["function"](sampled_vars=subs_dict, **init_step["inputs"])
        
        reaction_step = recipe[2]
        reaction_step["function"](state, **reaction_step["inputs"])
        
        output_path = graphics_dir / f"test_global_fbd_{i:04d}.png"
        generate_global_fbd(state, subs_dict, filepath=output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        generated_count += 1
        
    assert generated_count > 0, "No truss examples were found to run the test."
    print(f"\nGenerated {generated_count} Global FBDs for visual inspection.")


def test_generate_node_fbd_real_example():
    """Draws the isolated FBD of joint A for TRUSS_0001."""
    
    current_dir = Path(__file__).parent
    example_path = current_dir.parent / "tasks" / "example_TRUSS_0001.py"
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    if not example_path.exists():
        import pytest
        pytest.skip(f"Could not find {example_path}")
        
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    recipe = truss_module.SOLUTION_RECIPE
    subs_dict = truss_module.DRAWING_VARIABLES
    
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=subs_dict, **init_step["inputs"])
    
    reaction_step = recipe[2]
    reaction_step["function"](state, **reaction_step["inputs"])
    
    moj_step = recipe[3]
    moj_step["function"](state, **moj_step["inputs"])
    
    output_path = graphics_dir / "test_node_fbd_0001_A.png"
    generate_node_fbd(state, target_node_id="A", subs_dict=subs_dict, filepath=output_path)
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nGenerated Node FBD ready for inspection: {output_path}")


def test_generate_section_fbd_real_example():
    """Draws the method-of-sections FBD for TRUSS_0001."""
    current_dir = Path(__file__).parent
    example_path = current_dir.parent / "tasks" / "example_TRUSS_0001.py"
    graphics_dir = current_dir / "test_graphics"
    graphics_dir.mkdir(exist_ok=True)
    
    if not example_path.exists():
        import pytest
        pytest.skip(f"Could not find {example_path}")
        
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    recipe = truss_module.SOLUTION_RECIPE
    subs_dict = truss_module.DRAWING_VARIABLES
    
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=subs_dict, **init_step["inputs"])
    
    reaction_step = recipe[2]
    reaction_step["function"](state, **reaction_step["inputs"])
    
    moj_step = recipe[3]
    moj_step["function"](state, **moj_step["inputs"])
    
    output_path = graphics_dir / "test_section_fbd_0001_cut_peak.png"
    generate_section_fbd(
        state=state, 
        cut_element_ids=["BC", "CA"], 
        target_fbd_node="A", 
        subs_dict=subs_dict, 
        filepath=output_path
    )
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    print(f"\nGenerated Section FBD ready for inspection: {output_path}")
