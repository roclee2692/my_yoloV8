"""Strict structured evidence accepted by the Phase 11 report pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from people_flow.errors import ReportGenerationError

_EVALUATION_METRICS = (
    "id_switches",
    "enter_absolute_error",
    "exit_absolute_error",
    "total_count_absolute_error",
    "count_mae",
    "occupancy_mae",
)


class _StrictEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSummary(_StrictEvidence):
    """ROI occupancy and dwell summary emitted by the core pipeline."""

    roi_name: str = Field(min_length=1)
    total_enter: int = Field(ge=0)
    total_exit: int = Field(ge=0)
    maximum_occupancy: int = Field(ge=0)
    average_occupancy: float = Field(ge=0.0)
    peak_time: float | None = Field(default=None, ge=0.0)
    peak_frame_id: int | None = Field(default=None, ge=1)
    average_dwell_seconds: float = Field(ge=0.0)
    median_dwell_seconds: float = Field(ge=0.0)
    processed_frames: int = Field(gt=0)
    processing_fps: float = Field(gt=0.0)
    tracks_with_dwell: int = Field(ge=0)
    inside_at_end: int = Field(ge=0)


class RuntimeMetrics(_StrictEvidence):
    """Runtime metrics emitted by the detection and tracking pipeline."""

    model: str = Field(min_length=1)
    tracker: str = Field(min_length=1)
    device: str = Field(min_length=1)
    input_frames: int = Field(gt=0)
    processed_frames: int = Field(gt=0)
    video_fps: float = Field(gt=0.0)
    processing_fps: float = Field(gt=0.0)
    average_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    peak_gpu_memory_mb: float = Field(ge=0.0)
    total_runtime_seconds: float = Field(ge=0.0)


class EvaluationMetrics(_StrictEvidence):
    """Ground Truth availability and validated counting/tracking errors."""

    ground_truth_available: bool
    reason: str | None = None
    ground_truth_policy: dict[str, Any] | None = None
    tracking_iou_threshold: float = Field(gt=0.0, le=1.0)
    ground_truth_enter: int | None = Field(default=None, ge=0)
    ground_truth_exit: int | None = Field(default=None, ge=0)
    ground_truth_total: int | None = Field(default=None, ge=0)
    predicted_enter: int = Field(ge=0)
    predicted_exit: int = Field(ge=0)
    predicted_total: int = Field(ge=0)
    id_switches: int | None = Field(default=None, ge=0)
    enter_absolute_error: float | None = Field(default=None, ge=0.0)
    exit_absolute_error: float | None = Field(default=None, ge=0.0)
    total_count_absolute_error: float | None = Field(default=None, ge=0.0)
    count_mae: float | None = Field(default=None, ge=0.0)
    count_mape: float | None = Field(default=None, ge=0.0)
    occupancy_mae: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_ground_truth_state(self) -> EvaluationMetrics:
        """Prevent unavailable accuracy from being represented as numeric zero."""

        payload = self.model_dump()
        if not self.ground_truth_available:
            if not self.reason or not self.reason.strip():
                raise ValueError("Ground Truth unavailability requires a reason")
            fields = (
                "ground_truth_enter",
                "ground_truth_exit",
                "ground_truth_total",
                *_EVALUATION_METRICS,
                "count_mape",
            )
            populated = [field for field in fields if payload[field] is not None]
            if populated:
                raise ValueError(f"Ground Truth metrics must be null when unavailable: {populated}")
            return self
        required = (
            "ground_truth_enter",
            "ground_truth_exit",
            "ground_truth_total",
            *_EVALUATION_METRICS,
        )
        missing = [field for field in required if payload[field] is None]
        if missing:
            raise ValueError(f"Ground Truth metrics are missing: {missing}")
        return self


class ReportEvidence(_StrictEvidence):
    """The only three structured inputs visible to the report generator."""

    summary: RunSummary
    runtime_metrics: RuntimeMetrics
    evaluation: EvaluationMetrics

    @model_validator(mode="after")
    def reconcile_sources(self) -> ReportEvidence:
        """Reject contradictory frame counts and runtime throughput."""

        runtime = self.runtime_metrics
        if runtime.input_frames != runtime.processed_frames:
            raise ValueError("Report requires a complete run with no dropped input frames")
        if self.summary.processed_frames != runtime.processed_frames:
            raise ValueError("Summary and runtime processed frame counts do not match")
        if not math.isclose(
            self.summary.processing_fps,
            runtime.processing_fps,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("Summary and runtime processing FPS do not match")
        return self

    def prompt_payload(self) -> dict[str, Any]:
        """Return the bounded JSON object sent to an optional LLM."""

        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        """Return a stable SHA-256 fingerprint for report provenance."""

        canonical = json.dumps(
            self.prompt_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_report_evidence(run_dir: Path) -> ReportEvidence:
    """Load exactly summary, runtime metrics, and evaluation JSON files."""

    resolved = run_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise ReportGenerationError(f"Report run directory does not exist: {resolved}")
    paths = {
        "summary": resolved / "summary.json",
        "runtime_metrics": resolved / "runtime_metrics.json",
        "evaluation": resolved / "evaluation.json",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ReportGenerationError(
            f"Report requires summary.json, runtime_metrics.json, and evaluation.json; "
            f"missing {missing}: {resolved}"
        )
    try:
        return ReportEvidence(
            summary=RunSummary.model_validate(_read_json_object(paths["summary"])),
            runtime_metrics=RuntimeMetrics.model_validate(
                _read_json_object(paths["runtime_metrics"])
            ),
            evaluation=EvaluationMetrics.model_validate(_read_json_object(paths["evaluation"])),
        )
    except ValidationError as exc:
        raise ReportGenerationError(f"Structured report evidence is invalid: {exc}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportGenerationError(f"Unable to read report evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReportGenerationError(f"Report evidence must contain a JSON object: {path}")
    return payload
