"""Phase 2 import contract for the future line counter."""

from people_flow.counting import line_counter


def test_line_counter_scaffold_is_importable() -> None:
    """The line-counter boundary should import without model initialization."""

    assert line_counter.__doc__
