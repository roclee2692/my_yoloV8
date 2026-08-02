"""Validated Phase 8 experiment-matrix execution and comparison outputs."""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from people_flow.config import CountingSettings, RoiSettings, RunConfig, TrackerName, resolve_path
from people_flow.datasets.mot_ground_truth import (
    GroundTruthPolicy,
    ground_truth_track_records,
    load_mot_ground_truth,
)
from people_flow.errors import ConfigurationError, EvaluationError, OutputError
from people_flow.evaluation.tracking_metrics import compute_tracking_metrics
from people_flow.evaluation.trajectory_evaluator import evaluate_mot_prediction
from people_flow.evaluation.trajectory_io import read_track_records_csv
from people_flow.outputs.json_writer import write_json
from people_flow.pipeline import PipelineResult, run_pipeline
from people_flow.tracking.track_record import TrackRecord

_EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentSpec(_StrictModel):
    """One model/tracker combination in a controlled suite."""

    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    tracker: TrackerName

    @field_validator("id")
    @classmethod
    def validate_safe_id(cls, value: str) -> str:
        """Keep experiment IDs portable and prevent output path traversal."""

        if not _EXPERIMENT_ID_PATTERN.fullmatch(value):
            raise ValueError("experiment id must use only letters, digits, '.', '_' or '-'")
        return value


class GroundTruthSettings(_StrictModel):
    """Optional MOT Ground Truth association and matching policy."""

    mot_sequence: Path | None = None
    unavailable_reason: str = Field(
        default="The configured source has no associated Ground Truth.",
        min_length=1,
    )
    person_class_ids: tuple[int, ...] = (1,)
    require_marked: bool = True
    min_visibility: float = Field(default=0.0, ge=0.0, le=1.0)
    tracking_iou_threshold: float = Field(default=0.5, gt=0.0, le=1.0)

    @field_validator("unavailable_reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ground_truth.unavailable_reason must not be blank")
        return normalized

    def policy(self) -> GroundTruthPolicy:
        """Build the shared Phase 7 MOT annotation policy."""

        return GroundTruthPolicy(
            person_class_ids=self.person_class_ids,
            require_marked=self.require_marked,
            min_visibility=self.min_visibility,
        )


class ExperimentSuiteConfig(_StrictModel):
    """One controlled A/B/C experiment suite."""

    schema_version: Literal[1]
    suite_id: str = Field(min_length=1)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    source: Path
    output_root: Path = Path("runs/experiments")
    weights_dir: Path = Path("weights")
    device: str = Field(default="cpu", min_length=1)
    classes: tuple[int, ...] = (0,)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou: float = Field(default=0.7, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32, le=4096)
    counting: CountingSettings
    roi: RoiSettings
    ground_truth: GroundTruthSettings = Field(default_factory=GroundTruthSettings)
    experiments: tuple[ExperimentSpec, ...]

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str) -> str:
        if not _EXPERIMENT_ID_PATTERN.fullmatch(value):
            raise ValueError("suite_id must use only letters, digits, '.', '_' or '-'")
        return value

    @model_validator(mode="after")
    def validate_matrix(self) -> ExperimentSuiteConfig:
        if self.classes != (0,):
            raise ValueError("experiment suites only support COCO person class 0")
        if not self.counting.enabled:
            raise ValueError("Phase 8 suites require directional counting to be enabled")
        if not self.roi.enabled:
            raise ValueError("Phase 8 suites require ROI analysis to be enabled")
        if not self.experiments:
            raise ValueError("experiments must not be empty")
        ids = [experiment.id for experiment in self.experiments]
        if len(ids) != len(set(ids)):
            raise ValueError("experiment IDs must be unique")
        return self


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One stable row in the Phase 8 comparison table."""

    suite_id: str
    experiment_id: str
    seed: int
    source: str
    model: str
    tracker: str
    device: str
    confidence: float
    iou: float
    imgsz: int
    input_frames: int
    processed_frames: int
    track_observations: int
    unique_track_ids: int
    line_enter: int
    line_exit: int
    line_total: int
    roi_enter: int
    roi_exit: int
    maximum_occupancy: int
    average_occupancy: float
    processing_fps: float
    average_latency_ms: float
    p95_latency_ms: float
    peak_gpu_memory_mb: float
    total_runtime_seconds: float
    ground_truth_available: bool
    id_switches: int | None
    enter_absolute_error: int | None
    exit_absolute_error: int | None
    total_count_absolute_error: int | None
    count_mae: float | None
    count_mape: float | None
    occupancy_mae: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a CSV-compatible ordered mapping."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentSuiteResult:
    """Completed suite outputs and comparison rows."""

    output_root: Path
    comparison_csv: Path
    rows: tuple[ComparisonRow, ...]


PipelineRunner = Callable[[RunConfig], PipelineResult]


def load_experiment_suite(path: Path) -> ExperimentSuiteConfig:
    """Load a strict experiment-suite YAML file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"Experiment configuration does not exist: {resolved}")
    try:
        raw_data: Any = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Unable to read experiment configuration: {resolved}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid experiment YAML: {resolved}") from exc
    if not isinstance(raw_data, dict):
        raise ConfigurationError(f"Experiment configuration root must be a mapping: {resolved}")
    try:
        return ExperimentSuiteConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Experiment configuration validation failed: {resolved}\n{exc}"
        ) from exc


def run_experiment_suite(
    config: ExperimentSuiteConfig,
    *,
    base_dir: Path,
    overwrite: bool = False,
    pipeline_runner: PipelineRunner = run_pipeline,
) -> ExperimentSuiteResult:
    """Run every experiment with identical shared settings and aggregate outputs."""

    resolved_base = base_dir.expanduser().resolve()
    source = resolve_path(config.source, base_dir=resolved_base)
    if not source.is_file():
        raise EvaluationError(f"Experiment source does not exist: {source}")
    output_root = resolve_path(config.output_root, base_dir=resolved_base)
    comparison_path = output_root / "comparison.csv"
    if comparison_path.exists() and not overwrite:
        raise OutputError(f"Comparison output already exists: {comparison_path}. Use --overwrite.")

    rows: list[ComparisonRow] = []
    for experiment in config.experiments:
        output_dir = output_root / experiment.id
        _protect_experiment_directory(output_dir, overwrite=overwrite)
        _set_seed(config.seed, device=config.device)
        run_config = RunConfig(
            source=source,
            model=experiment.model,
            tracker=experiment.tracker,
            classes=config.classes,
            output_dir=output_dir,
            weights_dir=resolve_path(config.weights_dir, base_dir=resolved_base),
            device=config.device,
            confidence=config.confidence,
            iou=config.iou,
            imgsz=config.imgsz,
            overwrite=overwrite,
            counting=config.counting,
            roi=config.roi,
        )
        result = pipeline_runner(run_config)
        _verify_pipeline_outputs(result)
        _write_effective_config(output_dir / "config.yaml", config, experiment, result.run_config)
        predicted_records = read_track_records_csv(result.tracks_csv)
        evaluation_payload = _evaluation_payload(
            config,
            predicted_records,
            result,
            base_dir=resolved_base,
        )
        write_json(output_dir / "evaluation.json", evaluation_payload)
        rows.append(
            _comparison_row(
                config,
                experiment,
                source,
                result,
                predicted_records,
                evaluation_payload,
            )
        )

    _write_comparison_csv(comparison_path, rows)
    return ExperimentSuiteResult(output_root, comparison_path, tuple(rows))


def _set_seed(seed: int, *, device: str) -> None:
    """Set the available random sources before each controlled run."""

    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _protect_experiment_directory(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise OutputError(f"Experiment output is not empty: {output_dir}. Use --overwrite.")


def _verify_pipeline_outputs(result: PipelineResult) -> None:
    required = {
        "annotated.mp4": result.annotated_video,
        "tracks.csv": result.tracks_csv,
        "events.csv": result.events_csv,
        "occupancy.csv": result.occupancy_csv,
        "summary.json": result.summary_json,
        "runtime_metrics.json": result.runtime_metrics,
        "run.log": result.run_log,
    }
    missing = [name for name, path in required.items() if path is None or not path.is_file()]
    if missing:
        raise EvaluationError(f"Experiment pipeline did not create required artifacts: {missing}")


def _write_effective_config(
    path: Path,
    suite: ExperimentSuiteConfig,
    experiment: ExperimentSpec,
    persisted_run_config: Path,
) -> Path:
    try:
        run_payload: Any = yaml.safe_load(persisted_run_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OutputError(f"Unable to read pipeline run config: {persisted_run_config}") from exc
    if not isinstance(run_payload, dict):
        raise OutputError(f"Pipeline run config must contain a mapping: {persisted_run_config}")
    payload = {
        **run_payload,
        "suite_id": suite.suite_id,
        "experiment_id": experiment.id,
        "seed": suite.seed,
    }
    try:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"Unable to write effective experiment config: {path}") from exc
    return path


def _evaluation_payload(
    config: ExperimentSuiteConfig,
    predicted_records: tuple[TrackRecord, ...],
    result: PipelineResult,
    *,
    base_dir: Path,
) -> dict[str, Any]:
    ground_truth_settings = config.ground_truth
    if ground_truth_settings.mot_sequence is None:
        return {
            "ground_truth_available": False,
            "reason": ground_truth_settings.unavailable_reason,
            "ground_truth_policy": None,
            "tracking_iou_threshold": ground_truth_settings.tracking_iou_threshold,
            "ground_truth_enter": None,
            "ground_truth_exit": None,
            "ground_truth_total": None,
            "predicted_enter": result.total_enter,
            "predicted_exit": result.total_exit,
            "predicted_total": result.total_enter + result.total_exit,
            "id_switches": None,
            "enter_absolute_error": None,
            "exit_absolute_error": None,
            "total_count_absolute_error": None,
            "count_mae": None,
            "count_mape": None,
            "occupancy_mae": None,
        }

    mot_sequence = resolve_path(ground_truth_settings.mot_sequence, base_dir=base_dir)
    ground_truth = load_mot_ground_truth(mot_sequence)
    policy = ground_truth_settings.policy()
    counting = evaluate_mot_prediction(
        ground_truth,
        predicted_records,
        counting=config.counting,
        roi=config.roi,
        ground_truth_policy=policy,
    )
    tracking = compute_tracking_metrics(
        ground_truth_track_records(ground_truth, policy=policy),
        predicted_records,
        frame_count=ground_truth.sequence.sequence_length,
        iou_threshold=ground_truth_settings.tracking_iou_threshold,
    )
    return {
        **counting.as_dict(),
        **tracking.as_dict(),
        "mot_sequence": str(mot_sequence),
    }


def _comparison_row(
    suite: ExperimentSuiteConfig,
    experiment: ExperimentSpec,
    source: Path,
    result: PipelineResult,
    predicted_records: tuple[TrackRecord, ...],
    evaluation: dict[str, Any],
) -> ComparisonRow:
    summary = _read_json_object(result.summary_json)
    track_ids = {int(record.track_id) for record in predicted_records}
    return ComparisonRow(
        suite_id=suite.suite_id,
        experiment_id=experiment.id,
        seed=suite.seed,
        source=str(source),
        model=experiment.model,
        tracker=experiment.tracker,
        device=result.metrics.device,
        confidence=suite.confidence,
        iou=suite.iou,
        imgsz=suite.imgsz,
        input_frames=result.metrics.input_frames,
        processed_frames=result.metrics.processed_frames,
        track_observations=len(predicted_records),
        unique_track_ids=len(track_ids),
        line_enter=result.total_enter,
        line_exit=result.total_exit,
        line_total=result.total_enter + result.total_exit,
        roi_enter=result.roi_total_enter,
        roi_exit=result.roi_total_exit,
        maximum_occupancy=result.maximum_occupancy,
        average_occupancy=float(summary["average_occupancy"]),
        processing_fps=result.metrics.processing_fps,
        average_latency_ms=result.metrics.average_latency_ms,
        p95_latency_ms=result.metrics.p95_latency_ms,
        peak_gpu_memory_mb=result.metrics.peak_gpu_memory_mb,
        total_runtime_seconds=result.metrics.total_runtime_seconds,
        ground_truth_available=bool(evaluation["ground_truth_available"]),
        id_switches=evaluation["id_switches"],
        enter_absolute_error=evaluation["enter_absolute_error"],
        exit_absolute_error=evaluation["exit_absolute_error"],
        total_count_absolute_error=evaluation["total_count_absolute_error"],
        count_mae=evaluation["count_mae"],
        count_mape=evaluation["count_mape"],
        occupancy_mae=evaluation["occupancy_mae"],
    )


def _read_json_object(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise EvaluationError("Required summary JSON path is missing")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Unable to read experiment summary: {path}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"Experiment summary must contain a JSON object: {path}")
    return payload


def _write_comparison_csv(path: Path, rows: list[ComparisonRow]) -> Path:
    if not rows:
        raise EvaluationError("Cannot write an empty experiment comparison")
    fieldnames = list(rows[0].as_dict())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(row.as_dict() for row in rows)
    except OSError as exc:
        raise OutputError(f"Unable to write experiment comparison: {path}") from exc
    return path
