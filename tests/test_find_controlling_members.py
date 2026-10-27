import pytest
from trusstutor_bench.statics_tools import find_controlling_members


def test_tension_governs():
    """A fails in tension at load 5; B would only fail in compression at load 10."""
    load, state, members = find_controlling_members(
        coeffs={"A": 2.0, "B": -1.0}, limit_tension=10.0, limit_compression=10.0
    )
    assert load == pytest.approx(5.0)
    assert state == "tension"
    assert members == ["A"]


def test_compression_governs():
    """B fails in compression at load 2.5, before A fails in tension at load 10."""
    load, state, members = find_controlling_members(
        coeffs={"A": 1.0, "B": -4.0}, limit_tension=10.0, limit_compression=10.0
    )
    assert load == pytest.approx(2.5)
    assert state == "compression"
    assert members == ["B"]


def test_tied_members_are_all_returned_sorted():
    """A and B reach the limit at the same load, so both are returned in sorted order."""
    load, state, members = find_controlling_members(
        coeffs={"B": 2.0, "A": 2.0, "C": -1.0}, limit_tension=10.0, limit_compression=10.0
    )
    assert load == pytest.approx(5.0)
    assert state == "tension"
    assert members == ["A", "B"]


def test_offset_shifts_the_failure_load():
    """With a baseline force of 4, A reaches the limit at (10 - 4) / 2 = 3."""
    load, state, members = find_controlling_members(
        coeffs={"A": 2.0}, limit_tension=10.0, limit_compression=10.0, offsets={"A": 4.0}
    )
    assert load == pytest.approx(3.0)
    assert state == "tension"
    assert members == ["A"]


def test_member_already_over_limit_gives_zero_load():
    """A is over its tension limit by 50% and B by 20%, so A governs at load 0."""
    load, state, members = find_controlling_members(
        coeffs={"A": 1.0, "B": 1.0},
        limit_tension=10.0,
        limit_compression=10.0,
        offsets={"A": 15.0, "B": 12.0},
    )
    assert load == 0.0
    assert state == "tension"
    assert members == ["A"]


def test_already_over_limit_reports_the_worst_members_state():
    """A is 20% over in tension and B is 50% over in compression, so B and compression are reported."""
    load, state, members = find_controlling_members(
        coeffs={"A": 1.0, "B": 1.0},
        limit_tension=10.0,
        limit_compression=10.0,
        offsets={"A": 12.0, "B": -15.0},
    )
    assert load == 0.0
    assert state == "compression"
    assert members == ["B"]


def test_constant_member_at_the_limit_joins_the_tie():
    """B's force stays at 10 for any load, which equals the limit at the governing load."""
    load, state, members = find_controlling_members(
        coeffs={"A": 2.0, "B": 0.0},
        limit_tension=10.0,
        limit_compression=10.0,
        offsets={"B": 10.0},
    )
    assert load == pytest.approx(5.0)
    assert state == "tension"
    assert members == ["A", "B"]


def test_no_member_bounds_the_load_raises():
    """If no force depends on the load, no load can be found."""
    with pytest.raises(ValueError):
        find_controlling_members(
            coeffs={"A": 0.0}, limit_tension=10.0, limit_compression=10.0
        )
