"""Phase 2 import contract for the future MOT reader."""

from people_flow.datasets import mot_reader


def test_mot_reader_scaffold_is_importable() -> None:
    """The MOT-reader boundary should import without data downloads."""

    assert mot_reader.__doc__
