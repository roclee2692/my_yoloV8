"""Phase 2 import contract for the future geometry module."""

from people_flow.counting import geometry


def test_geometry_scaffold_is_importable() -> None:
    """The geometry package boundary should be available without optional dependencies."""

    assert geometry.__doc__
