"""Phase 2 import contract for future counting metrics."""

from people_flow.evaluation import counting_metrics


def test_counting_metrics_scaffold_is_importable() -> None:
    """The counting-metrics boundary should import without experiment data."""

    assert counting_metrics.__doc__
