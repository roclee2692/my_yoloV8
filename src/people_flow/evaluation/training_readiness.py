"""Conservative Phase 9 training-readiness assessment from baseline evidence."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from people_flow.errors import EvaluationError

_REQUIRED_COLUMNS = {
    "experiment_id",
    "ground_truth_available",
    "id_switches",
    "count_mae",
    "occupancy_mae",
}


@dataclass(frozen=True, slots=True)
class TrainingReadinessDecision:
    """Auditable decision on whether Phase 9 training may start."""

    decision: Literal["approved", "deferred"]
    training_allowed: bool
    source: str
    experiment_count: int
    ground_truth_evaluated_runs: int
    count_error_evaluated_runs: int
    detection_recall_evidence_available: bool
    detection_error_attribution_available: bool
    occlusion_failure_evidence_available: bool
    satisfied_trigger: str | None
    blockers: tuple[str, ...]
    next_action: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible decision mapping."""

        return asdict(self)


def assess_training_readiness(comparison_csv: Path) -> TrainingReadinessDecision:
    """Assess the current baseline without inferring unsupported training need.

    The Phase 8 comparison contains counting and tracking outcomes but does not yet
    contain detection Recall, error attribution, or occlusion-specific failure
    evidence. Consequently this assessor can approve nothing from those columns
    alone; it records the exact blockers that must be resolved first.
    """

    resolved = comparison_csv.expanduser().resolve()
    rows = _read_rows(resolved)
    ground_truth_runs = sum(_parse_bool(row["ground_truth_available"]) for row in rows)
    count_error_runs = sum(
        _parse_bool(row["ground_truth_available"])
        and _has_number(row["count_mae"])
        and _has_number(row["occupancy_mae"])
        for row in rows
    )

    blockers: list[str] = []
    if ground_truth_runs == 0:
        blockers.append("No baseline experiment has associated Ground Truth.")
    if count_error_runs == 0:
        blockers.append("No run has measured count and occupancy error.")
    blockers.extend(
        (
            "Detection Recall has not been measured on the evaluated source.",
            "No evidence attributes counting error primarily to missed detections.",
            "No occlusion-specific detection failure analysis is available.",
        )
    )
    return TrainingReadinessDecision(
        decision="deferred",
        training_allowed=False,
        source=str(resolved),
        experiment_count=len(rows),
        ground_truth_evaluated_runs=ground_truth_runs,
        count_error_evaluated_runs=count_error_runs,
        detection_recall_evidence_available=False,
        detection_error_attribution_available=False,
        occlusion_failure_evidence_available=False,
        satisfied_trigger=None,
        blockers=tuple(blockers),
        next_action=(
            "Evaluate the unchanged A/B/C matrix on an official MOT17 sequence, "
            "measure detection Recall, and attribute counting error before training."
        ),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise EvaluationError(f"Baseline comparison CSV does not exist: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_COLUMNS - fields)
            if missing:
                raise EvaluationError(
                    f"Baseline comparison is missing required columns {missing}: {path}"
                )
            rows = list(reader)
    except OSError as exc:
        raise EvaluationError(f"Unable to read baseline comparison CSV: {path}") from exc
    if not rows:
        raise EvaluationError(f"Baseline comparison contains no experiments: {path}")
    return rows


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise EvaluationError(f"Invalid ground_truth_available value: {value!r}")


def _has_number(value: str) -> bool:
    if not value.strip():
        return False
    try:
        float(value)
    except ValueError as exc:
        raise EvaluationError(f"Invalid numeric evaluation value: {value!r}") from exc
    return True
