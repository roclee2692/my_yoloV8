"""Phase 2 import contract for the future ROI counter."""

from people_flow.counting import roi_counter


def test_roi_counter_scaffold_is_importable() -> None:
    """The ROI-counter boundary should import without optional dependencies."""

    assert roi_counter.__doc__
