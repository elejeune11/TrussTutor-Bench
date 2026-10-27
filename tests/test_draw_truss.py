from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np
from numpy.testing import assert_allclose
import os
import pytest
import sympy as sp
from unittest.mock import patch

from trusstutor_bench.draw_truss import evaluate_geometry
from trusstutor_bench.draw_truss import calculate_bounding_box
from trusstutor_bench.draw_truss import _get_rotation_matrix
from trusstutor_bench.draw_truss import _transform_vertices
from trusstutor_bench.draw_truss import _draw_ground_plane
from trusstutor_bench.draw_truss import draw_pin
from trusstutor_bench.draw_truss import draw_roller
from trusstutor_bench.draw_truss import draw_elements_layer
from trusstutor_bench.draw_truss import draw_nodes_layer
from trusstutor_bench.draw_truss import draw_supports_layer
from trusstutor_bench.draw_truss import draw_loads_layer
from trusstutor_bench.draw_truss import generate_truss_diagram
from trusstutor_bench.draw_truss import draw_truss_adapter


def test_evaluate_geometry_string_parsing():
    """Tests the evaluation of textual string expressions."""
    nodes = {
        "A": (0, 0),
        "B": ("L", 0),
        "C": ("L / 2", "sqrt(3) * L / 2")
    }
    loads = {
        "C": (0, "-P")
    }
    subs_dict = {"L": 10.0, "P": 50.0}

    eval_nodes, eval_loads = evaluate_geometry(nodes, loads, subs_dict)

    assert eval_nodes["A"] == (0.0, 0.0)
    assert eval_nodes["B"] == (10.0, 0.0)
    assert eval_nodes["C"][0] == pytest.approx(5.0)
    assert eval_nodes["C"][1] == pytest.approx(8.660254037844386) # sqrt(3)*5
    
    assert eval_loads["C"] == (0.0, -50.0)

    assert type(eval_nodes["C"][1]) is float
    assert type(eval_loads["C"][1]) is float


def test_evaluate_geometry_sympy_objects():
    """Tests the evaluation of pre-instantiated SymPy variables."""
    L = sp.Symbol('L', positive=True)
    P = sp.Symbol('P')

    nodes = {
        "A": (0, L/2)
    }
    loads = {
        "A": (P/4, 0)
    }
    # SymPy's .subs() can handle a dictionary where the keys are the actual Symbols
    subs_dict = {L: 20.0, P: 100.0} 

    eval_nodes, eval_loads = evaluate_geometry(nodes, loads, subs_dict)

    assert eval_nodes["A"] == (0.0, 10.0)
    assert eval_loads["A"] == (25.0, 0.0)


def test_evaluate_geometry_pure_numbers():
    """Tests that pure numerical inputs are not corrupted by sympify."""
    nodes = {
        "A": (5, "15.5")
    }
    loads = {}
    subs_dict = {} # No substitutions needed

    eval_nodes, eval_loads = evaluate_geometry(nodes, loads, subs_dict)

    assert eval_nodes["A"] == (5.0, 15.5)
    assert type(eval_nodes["A"][0]) is float


def test_calculate_bounding_box_standard():
    """Tests a standard truss sitting entirely in the positive quadrant."""
    evaluated_nodes = {
        "A": (0.0, 0.0),
        "B": (10.0, 0.0),
        "C": (5.0, 8.66)
    }
    
    bbox = calculate_bounding_box(evaluated_nodes)
    
    assert bbox == (0.0, 10.0, 0.0, 8.66)


def test_calculate_bounding_box_with_negatives():
    """Tests a truss that extends into negative coordinates (e.g., a cantilever)."""
    evaluated_nodes = {
        "A": (0.0, 0.0),
        "B": (-5.0, -5.0),
        "C": (10.5, -2.0)
    }
    
    bbox = calculate_bounding_box(evaluated_nodes)
    
    # min_x, max_x, min_y, max_y
    assert bbox == (-5.0, 10.5, -5.0, 0.0)


def test_calculate_bounding_box_single_node():
    """Tests an edge case where a 'truss' only has one node to ensure it doesn't crash."""
    evaluated_nodes = {
        "Origin": (5.0, 5.0)
    }
    
    bbox = calculate_bounding_box(evaluated_nodes)
    
    # If there's only one node, the min and max should be identical
    assert bbox == (5.0, 5.0, 5.0, 5.0)


def test_get_rotation_matrix():
    """Tests standard 2D rotation matrices for 0, 90, 180, and -90 degrees."""
    
    # 0 degrees: The Identity Matrix (No rotation)
    R_0 = _get_rotation_matrix(0)
    expected_0 = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    assert_allclose(R_0, expected_0, atol=1e-9)

    # 90 degrees: Counter-clockwise quarter turn
    R_90 = _get_rotation_matrix(90)
    expected_90 = np.array([
        [0.0, -1.0],
        [1.0,  0.0]
    ])
    assert_allclose(R_90, expected_90, atol=1e-9)

    # 180 degrees: Half turn (flips both X and Y)
    R_180 = _get_rotation_matrix(180)
    expected_180 = np.array([
        [-1.0,  0.0],
        [ 0.0, -1.0]
    ])
    assert_allclose(R_180, expected_180, atol=1e-9)

    # -90 degrees: Clockwise quarter turn (wall support on the left edge)
    R_neg90 = _get_rotation_matrix(-90)
    expected_neg90 = np.array([
        [ 0.0, 1.0],
        [-1.0, 0.0]
    ])
    assert_allclose(R_neg90, expected_neg90, atol=1e-9)


def test_transform_vertices_translation_only():
    """Tests that local coordinates are moved correctly when angle is 0."""
    local_shape = [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0)]
    
    # Move to X=10, Y=5 with no rotation
    result = _transform_vertices(local_shape, global_x=10.0, global_y=5.0, angle_degrees=0.0)
    
    expected = [
        (10.0, 5.0), 
        (11.0, 5.0), 
        (10.0, 7.0)
    ]
    assert_allclose(np.array(result), np.array(expected), atol=1e-9)


def test_transform_vertices_rotation_only():
    """Tests that a shape rotates perfectly around the origin."""
    local_shape = [(1.0, 0.0), (0.0, 1.0)]
    
    # Rotate 90 degrees counter-clockwise, keep at origin
    result = _transform_vertices(local_shape, global_x=0.0, global_y=0.0, angle_degrees=90.0)
    
    expected = [
        (0.0, 1.0),   # (1,0) swings up to the Y axis
        (-1.0, 0.0)   # (0,1) swings left to the negative X axis
    ]
    assert_allclose(np.array(result), np.array(expected), atol=1e-9)


def test_transform_vertices_combined():
    """Tests the combined rotation and translation of a wall support."""
    # A generic local triangle pointing "up"
    local_triangle = [(0.0, 0.0), (-1.0, -1.0), (1.0, -1.0)]
    
    # Rotate -90 degrees (wall on the left, pointing right) and move to (5, 5)
    result = _transform_vertices(local_triangle, global_x=5.0, global_y=5.0, angle_degrees=-90.0)
    
    # Expected math:
    # (0,0) rotated is (0,0) -> translated to (5,5)
    # (-1,-1) rotated -90 is (-1, 1) -> translated to (4,6)
    # (1,-1) rotated -90 is (-1, -1) -> translated to (4,4)
    expected = [
        (5.0, 5.0),
        (4.0, 6.0),
        (4.0, 4.0)
    ]
    assert_allclose(np.array(result), np.array(expected), atol=1e-9)


def test_evaluate_geometry_loads_comprehensive():
    """Evaluates loads given as strings, SymPy objects and plain numbers into floats."""
    P = sp.Symbol('P')
    
    # Minimal nodes dict to satisfy the first half of the function
    nodes = {"A": (0, 0)} 
    
    # Testing all three input types the function might encounter
    loads = {
        "Node1": (0, "-P"),           # Pure strings
        "Node2": (P / 2.0, 0),        # SymPy objects
        "Node3": (15, "-30.5")        # Integers and string floats
    }
    
    subs_dict = {"P": 50.0}
    
    _, eval_loads = evaluate_geometry(nodes, loads, subs_dict)
    
    assert eval_loads["Node1"] == (0.0, -50.0)
    assert eval_loads["Node2"] == (25.0, 0.0)
    assert eval_loads["Node3"] == (15.0, -30.5)
    
    for nid, (fx, fy) in eval_loads.items():
        assert type(fx) is float, f"X-force on {nid} is not a float"
        assert type(fy) is float, f"Y-force on {nid} is not a float"


def test_calculate_bounding_box_empty():
    """Tests the early exit condition when the nodes dictionary is empty."""
    evaluated_nodes = {}
    
    bbox = calculate_bounding_box(evaluated_nodes)
    
    # Should safely return all zeros without throwing a ValueError from min()/max()
    assert bbox == (0.0, 0.0, 0.0, 0.0)


def test_draw_ground_plane():
    """Draws the expected number of ground-plane lines, in the right place."""
    # Create a dummy figure and axis for testing
    fig, ax = plt.subplots()
    
    # Call the function at the origin with no rotation
    _draw_ground_plane(ax, center_x=0.0, center_y=0.0, angle_degrees=0.0, scale=1.0)
    
    lines = ax.get_lines()
    assert len(lines) > 2, "Ground plane should generate a main line and multiple hashes"
    
    main_line = lines[0]
    x_data = main_line.get_xdata()
    y_data = main_line.get_ydata()
    
    # Width is 0.8, centered at 0, so it should span from -0.4 to 0.4
    assert_allclose(x_data, [-0.4, 0.4], atol=1e-9)
    assert_allclose(y_data, [0.0, 0.0], atol=1e-9)
    
    assert main_line.get_linewidth() == 2.5
    assert lines[1].get_linewidth() == 1.0 # The first hash mark
    
    # Clean up Matplotlib memory so tests run fast
    plt.close(fig)


def test_draw_ground_plane_wall():
    """
    Verifies that the ground plane correctly orientates when rotated for a wall support.
    """
    fig, ax = plt.subplots()
    
    # Call the function at X=5, Y=10, rotated 90 degrees (right-facing wall)
    _draw_ground_plane(ax, center_x=5.0, center_y=10.0, angle_degrees=90.0, scale=1.0)
    
    lines = ax.get_lines()
    main_line = lines[0]
    x_data = main_line.get_xdata()
    y_data = main_line.get_ydata()
    
    # The line was originally horizontal (-0.4 to 0.4 on X). 
    # Rotated 90 degrees, it should be vertical on the Y axis, centered at X=5.
    assert_allclose(x_data, [5.0, 5.0], atol=1e-9)
    assert_allclose(y_data, [9.6, 10.4], atol=1e-9)
    
    plt.close(fig)


def test_draw_pin():
    """Draws the pin as a polygon anchored at the node, with the ground-plane lines."""
    fig, ax = plt.subplots()
    
    # Call the pin at (10, 10) with no rotation
    draw_pin(ax, x=10.0, y=10.0, angle=0.0, scale=1.0)
    
    assert len(ax.patches) == 1, "Pin should generate exactly one Polygon patch"
    pin_patch = ax.patches[0]
    assert isinstance(pin_patch, patches.Polygon)
    
    # Check that the top vertex snapped perfectly to the node coordinate (10, 10)
    vertices = pin_patch.get_xy()
    assert_allclose(vertices[0], [10.0, 10.0], atol=1e-9)
    
    lines = ax.get_lines()
    assert len(lines) > 2, "Pin should have called _draw_ground_plane to generate hashes"
    plt.close(fig)


def test_draw_pin_rotated():
    """
    Verifies that the pin correctly maps to a wall when rotated 90 degrees.
    """
    fig, ax = plt.subplots()
    
    draw_pin(ax, x=0.0, y=5.0, angle=90.0, scale=1.0)
    
    pin_patch = ax.patches[0]
    vertices = pin_patch.get_xy()
    
    # The top vertex should still be at (0, 5)
    assert_allclose(vertices[0], [0.0, 5.0], atol=1e-9)
    
    # A standard pin has its base at Y = -h. 
    # Rotated 90 degrees CCW, the base should swing to X = h.
    # We check the X-coordinate of the second vertex (originally bottom-left).
    # Since w=0.6, h=0.5196...
    expected_h = 0.6 * np.sqrt(3) / 2.0
    assert_allclose(vertices[1][0], expected_h, atol=1e-9)
    plt.close(fig)


def test_draw_roller_patches():
    """Draws the roller as one triangle and two wheels."""
    fig, ax = plt.subplots()
    
    draw_roller(ax, x=5.0, y=5.0, angle=0.0, scale=1.0)
    
    assert len(ax.patches) == 3, "Roller should generate 3 patches (1 triangle, 2 circles)"
    
    # Matplotlib stores patches in the order they were added
    triangle = ax.patches[0]
    wheel_1 = ax.patches[1]
    wheel_2 = ax.patches[2]
    
    assert isinstance(triangle, patches.Polygon)
    assert isinstance(wheel_1, patches.Circle)
    assert isinstance(wheel_2, patches.Circle)
    
    # Verify the triangle snapped perfectly to the node
    vertices = triangle.get_xy()
    assert_allclose(vertices[0], [5.0, 5.0], atol=1e-9)
    plt.close(fig)


def test_draw_roller_ground_offset():
    """
    Verifies the roller circles and ground plane correctly account for the radius offset.
    """
    fig, ax = plt.subplots()
    scale = 1.0
    r = 0.1 * scale
    w = 0.6 * scale
    h = w * np.sqrt(3) / 2.0
    
    draw_roller(ax, x=0.0, y=0.0, angle=0.0, scale=scale)
    
    wheel_1 = ax.patches[1]
    
    # Wheel centers should be at Y = -h - r
    expected_y = -h - r
    assert_allclose(wheel_1.center[1], expected_y, atol=1e-9)
    
    # Ground plane (the main baseline) should be at Y = -h - 2*r
    lines = ax.get_lines()
    main_baseline = lines[0]
    ground_y_data = main_baseline.get_ydata()
    
    expected_ground_y = -h - 2 * r
    assert_allclose(ground_y_data[0], expected_ground_y, atol=1e-9)
    
    plt.close(fig)


def test_draw_roller_rotated():
    """
    Verifies the entire assembly successfully maps to a vertical wall.
    """
    fig, ax = plt.subplots()
    
    # Node at origin, rotate 90 degrees (wall on right)
    draw_roller(ax, x=0.0, y=0.0, angle=90.0, scale=1.0)
    
    # Fetch the first wheel
    wheel_1 = ax.patches[1]
    
    # Originally at Y = -h - r. 
    # Rotated 90 deg CCW, the -Y offset becomes an +X offset.
    scale = 1.0
    w = 0.6 * scale
    h = w * np.sqrt(3) / 2.0
    r = 0.1 * scale
    expected_x = h + r
    
    assert_allclose(wheel_1.center[0], expected_x, atol=1e-9)
    
    plt.close(fig)


def test_draw_elements_layer_default_no_labels():
    """Draws the members without labels by default."""
    fig, ax = plt.subplots()
    
    evaluated_nodes = {"A": (0.0, 0.0), "B": (10.0, 0.0)}
    elements = {"1": ("A", "B")}
    
    # Do not pass the show_labels flag (should default to False)
    draw_elements_layer(ax, evaluated_nodes, elements)
    
    assert len(ax.get_lines()) == 1
    
    assert len(ax.texts) == 0, "Labels should be hidden by default"
    
    plt.close(fig)


def test_draw_elements_layer_with_labels():
    """With show_labels=True, the labels are centered at the member midpoints."""
    fig, ax = plt.subplots()
    
    evaluated_nodes = {
        "A": (0.0, 0.0),
        "B": (10.0, 0.0),
        "C": (5.0, 8.66)
    }
    
    elements = {
        "1": ("A", "B"),
        "2": ("B", "C")
    }
    
    # Explicitly turn labels on
    draw_elements_layer(ax, evaluated_nodes, elements, show_labels=True)
    
    lines = ax.get_lines()
    texts = ax.texts
    
    assert len(lines) == 2
    assert len(texts) == 2, "Should generate one text label per element"
    
    # Verify the midpoint math and label data for Element 1
    text_1 = texts[0]
    assert text_1.get_text() == "1"
    assert_allclose(text_1.get_position(), [5.0, 0.0], atol=1e-9)
    plt.close(fig)


def test_draw_nodes_layer():
    """Draws one marker and one label per node, with the right style and position."""
    fig, ax = plt.subplots()
    
    evaluated_nodes = {
        "A": (0.0, 0.0),
        "B": (10.0, 5.0)
    }
    
    draw_nodes_layer(ax, evaluated_nodes)
    
    lines = ax.get_lines()
    assert len(lines) == 2, "Should draw one marker Line2D per node"
    
    # Verify the marker properties for Node A
    node_A_marker = lines[0]
    assert node_A_marker.get_marker() == 'o'
    assert node_A_marker.get_markerfacecolor() == 'white'
    assert_allclose(node_A_marker.get_xdata(), [0.0], atol=1e-9)
    assert_allclose(node_A_marker.get_ydata(), [0.0], atol=1e-9)
    
    texts = ax.texts
    assert len(texts) == 2, "Should draw one text annotation per node"
    
    # Verify the text content and position for Node B
    text_B = texts[1]
    assert text_B.get_text() == "B"
    
    # In Matplotlib, ax.texts stores annotations. 
    # .xy is the coordinate being annotated, .get_position() is the offset.
    assert_allclose(text_B.xy, [10.0, 5.0], atol=1e-9)
    plt.close(fig)


# @patch intercepts the component painters so we can check the arguments 
# passed to them without actually rendering the Matplotlib patches.
@patch('trusstutor_bench.draw_truss.draw_pin')
@patch('trusstutor_bench.draw_truss.draw_roller')
def test_draw_supports_layer_bridge(mock_draw_roller, mock_draw_pin):
    """Tests a standard bridge where supports are spread out on the X-axis."""
    fig, ax = plt.subplots()
    
    # A standard triangular bridge
    evaluated_nodes = {"A": (0.0, 0.0), "B": (10.0, 0.0), "C": (5.0, 5.0)}
    supports = {"A": ['x', 'y'], "B": ['y']}  # Pin at A, Floor Roller at B
    
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0)
    
    # Assert Pin defaults to the floor (0.0)
    mock_draw_pin.assert_called_once_with(ax, 0.0, 0.0, angle=0.0, scale=1.0)
    
    # Assert Roller defaults to the floor (0.0)
    mock_draw_roller.assert_called_once_with(ax, 10.0, 0.0, angle=0.0, scale=1.0)
    
    plt.close(fig)

@patch('trusstutor_bench.draw_truss.draw_pin')
def test_draw_supports_layer_left_cantilever(mock_draw_pin):
    """Tests that a truss with all supports on the left edge mounts to a wall."""
    fig, ax = plt.subplots()
    
    # Nodes stretching out to the right (X=10)
    evaluated_nodes = {"A": (0.0, 0.0), "B": (0.0, 10.0), "C": (10.0, 5.0)}
    supports = {"A": ['x', 'y'], "B": ['x', 'y']}  # Two pins on the left wall
    
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0)
    
    # Both pins should rotate -90 degrees to point to the right
    mock_draw_pin.assert_any_call(ax, 0.0, 0.0, angle=-90.0, scale=1.0)
    mock_draw_pin.assert_any_call(ax, 0.0, 10.0, angle=-90.0, scale=1.0)
    assert mock_draw_pin.call_count == 2
    
    plt.close(fig)

@patch('trusstutor_bench.draw_truss.draw_pin')
def test_draw_supports_layer_right_cantilever(mock_draw_pin):
    """Tests that a truss with all supports on the right edge mounts correctly."""
    fig, ax = plt.subplots()
    
    # Nodes stretching out to the left (X=0), supported at X=10
    evaluated_nodes = {"A": (10.0, 0.0), "B": (10.0, 10.0), "C": (0.0, 5.0)}
    supports = {"A": ['x', 'y'], "B": ['x', 'y']}  
    
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0)
    
    # Both pins should rotate 90 degrees to point to the left
    mock_draw_pin.assert_any_call(ax, 10.0, 0.0, angle=90.0, scale=1.0)
    mock_draw_pin.assert_any_call(ax, 10.0, 10.0, angle=90.0, scale=1.0)
    assert mock_draw_pin.call_count == 2
    plt.close(fig)

@patch('trusstutor_bench.draw_truss.draw_pin')
def test_draw_supports_layer_ceiling(mock_draw_pin):
    """Tests that a hanging truss detects the ceiling and flips the pins 180 degrees."""
    fig, ax = plt.subplots()
    
    # Supports are spread out at Y=10, the rest of the truss hangs down to Y=0
    evaluated_nodes = {"A": (0.0, 10.0), "B": (10.0, 10.0), "C": (5.0, 0.0)}
    supports = {"A": ['x', 'y'], "B": ['x', 'y']} 
    
    draw_supports_layer(ax, evaluated_nodes, supports, scale_factor=1.0)
    
    # Both pins should rotate 180 degrees to point downward
    mock_draw_pin.assert_any_call(ax, 0.0, 10.0, angle=180.0, scale=1.0)
    mock_draw_pin.assert_any_call(ax, 10.0, 10.0, angle=180.0, scale=1.0)
    assert mock_draw_pin.call_count == 2
    
    plt.close(fig)


def test_draw_loads_layer():
    """Draws load arrows at the nodes, skips zero loads and prefers symbolic labels."""
    fig, ax = plt.subplots()
    
    evaluated_nodes = {
        "A": (0.0, 0.0),
        "B": (10.0, 0.0)
    }
    
    evaluated_loads = {
        "A": (0.0, -50.0),   # 50 units pointing straight down
        "B": (0.0, 0.0)      # No force, should be completely skipped
    }
    
    # Original symbolic loads, used for the labels.
    original_loads = {
        "A": (0, "-P"),
        "B": (0, 0)
    }
    
    # Pass the new original_loads parameter
    draw_loads_layer(ax, evaluated_nodes, evaluated_loads, original_loads, scale_factor=1.0)
    
    texts = ax.texts
    assert len(texts) == 2, "Should generate exactly one arrow and one label for Node A."
    
    annotations = [t for t in texts if isinstance(t, Annotation)]
    standard_texts = [t for t in texts if not isinstance(t, Annotation)]
    
    assert len(annotations) == 1
    assert len(standard_texts) == 1
    
    arrow = annotations[0]
    label = standard_texts[0]
    
    assert_allclose(arrow.xy, [0.0, 0.0], atol=1e-9)             # Tip exactly at node
    assert_allclose(arrow.get_position(), [0.0, 1.5], atol=1e-9) # Tail floating above
    
    # Verify color assignment
    expected_rgba = to_rgba('#cc0000')
    assert arrow.arrow_patch.get_facecolor() == expected_rgba
    assert arrow.arrow_patch.get_edgecolor() == expected_rgba
    
    assert label.get_text() == "P"
    assert label.get_color() == '#cc0000'
    
    # The text is offset by 20% further out from the tail. (1.5 * 1.2 = 1.8)
    assert_allclose(label.get_position(), [0.0, 1.8], atol=1e-9)
    
    plt.close(fig)


def test_generate_truss_diagram_integration(tmp_path):
    """Draws a symbolic truss end to end and saves a high-resolution PNG."""
    # Define a complete, simple bridge with symbolic variables
    nodes = {
        "A": (0, 0),
        "B": ("L", 0),
        "C": ("L/2", "H")
    }
    
    elements = {
        "1": ("A", "B"),
        "2": ("B", "C"),
        "3": ("A", "C")
    }
    
    loads = {
        "C": (0, "-P")
    }
    
    supports = {
        "A": ['x', 'y'],  # Pin
        "B": ['y']        # Roller
    }
    
    subs_dict = {
        "L": 10.0,
        "H": 8.66,
        "P": 500.0
    }
    
    # Use the pytest tmp_path fixture to generate a safe output destination
    output_file = tmp_path / "test_truss.png"
    
    # Execute the master function
    generate_truss_diagram(
        nodes, 
        elements, 
        loads, 
        supports, 
        subs_dict, 
        filepath=str(output_file), 
        show_labels=True
    )
    
    # Assertions
    assert output_file.exists(), "The PNG file was not generated."
    
    file_size = os.path.getsize(output_file)
    assert file_size > 0, "The generated PNG file is empty."


@pytest.fixture
def sample_truss_data():
    """
    Provides the TRUSS_0001 equilateral truss data (nodes, elements, loads, supports, metadata and substitutions).

    Returns:
        dict: The data, passed to the test as a fixture.
    """
    L = sp.Symbol('L', positive=True)
    P = sp.Symbol('P')
    
    nodes = {
        "A": (0, 0),
        "B": (L, 0),
        "C": (sp.Rational(1, 2) * L, sp.sqrt(3) * L / 2)
    }
    
    elements = [
        {'id': 'AB', 'nodes': ('A', 'B'), 'E': 1.0, 'A': 1.0},
        {'id': 'BC', 'nodes': ('B', 'C'), 'E': 1.0, 'A': 1.0},
        {'id': 'CA', 'nodes': ('C', 'A'), 'E': 1.0, 'A': 1.0}
    ]
    
    loads = {
        "C": (0, -P)
    }
    
    supports = {
        "A": ['x', 'y'],
        "B": ['y']
    }
    
    metadata = {
        "benchmark_id": "TRUSS_0001"
    }
    
    subs_dict = {
        "L": 10.0, 
        "P": 50.0
    }

    return nodes, elements, loads, supports, metadata, subs_dict


def test_draw_truss_adapter_creates_file(sample_truss_data, tmp_path):
    """The adapter creates the output folder and writes a non-empty image."""
    # Unpack the fixture data
    nodes, elements, loads, supports, metadata, subs_dict = sample_truss_data
    
    # tmp_path is a built-in pytest fixture providing a unique temporary directory path.
    # We create a sub-folder explicitly to test the os.makedirs logic in the adapter.
    test_output_dir = tmp_path / "test_outputs"
    
    draw_truss_adapter(
        nodes=nodes,
        elements_list=elements,
        loads=loads,
        supports=supports,
        metadata=metadata,
        subs_dict=subs_dict,
        output_dir=str(test_output_dir)
    )
    
    expected_filename = f"{metadata['benchmark_id']}_truss_schematic.png"
    expected_filepath = test_output_dir / expected_filename
    
    assert test_output_dir.exists(), "Adapter failed to create the target directory."
    assert expected_filepath.exists(), f"Adapter failed to save {expected_filename}."
    assert expected_filepath.stat().st_size > 0, "The generated image file is 0 bytes (empty)."
