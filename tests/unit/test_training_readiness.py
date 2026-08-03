"""Tests for the conservative Phase 9 training gate."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from people_flow.errors import EvaluationError
from people_flow.evaluation.training_readiness import assess_training_readiness


def _write_comparison(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "experiment_id",
        "ground_truth_available",
        "id_switches",
        "count_mae",
        "occupancy_mae",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_training_is_deferred_without_ground_truth(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    _write_comparison(
        comparison,
        [
            {
                "experiment_id": "A",
                "ground_truth_available": "False",
                "id_switches": "",
                "count_mae": "",
                "occupancy_mae": "",
            }
        ],
    )

    decision = assess_training_readiness(comparison)

    assert decision.decision == "deferred"
    assert decision.training_allowed is False
    assert decision.ground_truth_evaluated_runs == 0
    assert decision.count_error_evaluated_runs == 0
    assert decision.satisfied_trigger is None
    assert any("Ground Truth" in blocker for blocker in decision.blockers)


def test_count_errors_alone_do_not_prove_finetuning_is_needed(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    _write_comparison(
        comparison,
        [
            {
                "experiment_id": "A",
                "ground_truth_available": "True",
                "id_switches": "1",
                "count_mae": "2.0",
                "occupancy_mae": "0.5",
            }
        ],
    )

    decision = assess_training_readiness(comparison)

    assert decision.ground_truth_evaluated_runs == 1
    assert decision.count_error_evaluated_runs == 1
    assert decision.training_allowed is False
    assert decision.detection_error_attribution_available is False
    assert any("attributes counting error" in blocker for blocker in decision.blockers)


def test_training_gate_rejects_missing_schema(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    comparison.write_text("experiment_id\nA\n", encoding="utf-8")

    with pytest.raises(EvaluationError, match="missing required columns"):
        assess_training_readiness(comparison)
