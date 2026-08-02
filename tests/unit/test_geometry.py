"""Tests for directed-line side and finite-segment geometry."""

import pytest

from people_flow.counting.geometry import DirectedLine, LineSide, segments_intersect
from people_flow.errors import CountingError


def test_directed_line_classifies_positive_negative_and_on_line() -> None:
    """Side signs must follow the p1-to-p2 direction."""

    line = DirectedLine((0.0, 0.0), (10.0, 0.0))

    assert line.side_of((5.0, 3.0)) is LineSide.POSITIVE
    assert line.side_of((5.0, -2.0)) is LineSide.NEGATIVE
    assert line.side_of((5.0, 0.0)) is LineSide.ON_LINE


def test_reversing_line_endpoints_reverses_side_sign() -> None:
    """Users can reverse direction by swapping line endpoints."""

    forward = DirectedLine((0.0, 0.0), (10.0, 0.0))
    reversed_line = DirectedLine((10.0, 0.0), (0.0, 0.0))

    assert forward.side_of((5.0, 2.0)) is LineSide.POSITIVE
    assert reversed_line.side_of((5.0, 2.0)) is LineSide.NEGATIVE


def test_motion_must_cross_the_finite_line_segment() -> None:
    """Crossing the infinite extension outside the endpoints must not count."""

    line = DirectedLine((0.0, 0.0), (10.0, 0.0))

    assert line.intersects_motion((5.0, -2.0), (5.0, 2.0))
    assert not line.intersects_motion((20.0, -2.0), (20.0, 2.0))
    assert segments_intersect((10.0, -2.0), (10.0, 0.0), line.p1, line.p2)


def test_enter_side_can_be_swapped_without_changing_geometry() -> None:
    """The destination side determines whether the event is enter or exit."""

    line = DirectedLine((0.0, 0.0), (10.0, 0.0), enter_side=LineSide.NEGATIVE)

    assert line.event_type_for(LineSide.NEGATIVE) == "enter"
    assert line.event_type_for(LineSide.POSITIVE) == "exit"


def test_zero_length_line_is_rejected() -> None:
    """A line with identical endpoints has no meaningful side."""

    with pytest.raises(CountingError, match="must be different"):
        DirectedLine((1.0, 1.0), (1.0, 1.0))
