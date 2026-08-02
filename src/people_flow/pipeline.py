"""Person tracking and directional people-counting pipeline."""

from __future__ import annotations

import platform
import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import cv2
import yaml

from people_flow import __version__
from people_flow.assets import resolve_model_path
from people_flow.config import RunConfig, resolve_path
from people_flow.counting.geometry import DirectedLine, LineSide
from people_flow.counting.line_counter import LineCounter
from people_flow.datasets.video_source import Frame, VideoSource
from people_flow.detection.yolo_detector import YoloDetector
from people_flow.errors import OutputError, PipelineError
from people_flow.evaluation.runtime_metrics import (
    RuntimeMetrics,
    build_runtime_metrics,
    read_peak_gpu_memory_mb,
    reset_peak_gpu_memory,
)
from people_flow.logging import configure_logging
from people_flow.outputs.csv_writer import TrackCsvWriter
from people_flow.outputs.event_writer import EventCsvWriter
from people_flow.outputs.evidence_writer import write_evidence_frame
from people_flow.outputs.json_writer import write_json
from people_flow.outputs.video_writer import AnnotatedVideoWriter
from people_flow.tracking.track_record import TrackRecord
from people_flow.tracking.tracker_runner import TrackerRunner
from people_flow.visualization.annotator import annotate_frame

_OUTPUT_NAMES = (
    "annotated.mp4",
    "tracks.csv",
    "run_config.yaml",
    "events.csv",
    "runtime_metrics.json",
    "run.log",
)


class FrameTracker(Protocol):
    """Pipeline-facing tracker contract for real and test implementations."""

    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        """Return all person tracks for one frame."""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Paths and metrics produced by a successful pipeline run."""

    output_dir: Path
    annotated_video: Path
    tracks_csv: Path
    run_config: Path
    events_csv: Path | None
    evidence_dir: Path | None
    runtime_metrics: Path
    run_log: Path
    metrics: RuntimeMetrics

    total_enter: int
    total_exit: int


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> Path:
    """Create the output directory while preserving unrelated files."""

    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    existing = [resolved / name for name in _OUTPUT_NAMES if (resolved / name).exists()]
    evidence_dir = resolved / "evidence"
    if evidence_dir.is_dir() and any(evidence_dir.iterdir()):
        existing.append(evidence_dir)
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise OutputError(
            f"Output files already exist in {resolved}: {names}. Use --overwrite to replace them."
        )
    if overwrite and evidence_dir.exists():
        try:
            evidence_dir.relative_to(resolved)
            shutil.rmtree(evidence_dir)
        except OSError as exc:
            raise OutputError(f"Unable to clear evidence directory: {evidence_dir}") from exc
    return resolved


def _write_run_config(
    path: Path,
    *,
    config: RunConfig,
    source: Path,
    model_path: Path,
) -> Path:
    """Persist the effective run arguments and software versions as YAML."""

    data = {
        "schema_version": 1,
        "source": str(source),
        "model": config.model,
        "resolved_model": str(model_path),
        "tracker": config.tracker,
        "classes": list(config.classes),
        "output_dir": str(path.parent),
        "weights_dir": str(config.weights_dir.expanduser().resolve()),
        "device": config.device,
        "confidence": config.confidence,
        "iou": config.iou,
        "imgsz": config.imgsz,
        "counting": config.counting.model_dump(mode="json"),
        "software": {
            "people_flow": __version__,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "torch": version("torch"),
            "ultralytics": version("ultralytics"),
        },
    }
    try:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        raise OutputError(f"Unable to write run configuration: {path}") from exc
    return path


def _build_line_counter(config: RunConfig) -> LineCounter | None:
    """Build a counter only when explicit line geometry enables counting."""

    settings = config.counting
    if not settings.enabled:
        return None
    if settings.line is None:
        raise PipelineError("Counting is enabled but no counting line was configured")
    directed_line = DirectedLine(
        p1=settings.line.p1,
        p2=settings.line.p2,
        enter_side=LineSide(settings.line.enter_side),
    )
    return LineCounter(
        directed_line,
        min_track_age=settings.min_track_age,
        min_displacement_pixels=settings.min_displacement_pixels,
        cooldown_frames=settings.cooldown_frames,
        max_track_gap_frames=settings.max_track_gap_frames,
    )


def run_pipeline(
    config: RunConfig, *, tracker_runner: FrameTracker | None = None
) -> PipelineResult:
    """Run person-only tracking and optional directional counting on a local video."""

    source = resolve_path(config.source, base_dir=Path.cwd())
    output_dir = _prepare_output_directory(config.output_dir, overwrite=config.overwrite)
    run_log = output_dir / "run.log"
    logger = configure_logging(level="INFO", log_file=run_log)
    logger.info("Starting people-flow run: source=%s model=%s", source, config.model)
    total_start = time.perf_counter()

    model_path = resolve_model_path(config.model, weights_dir=config.weights_dir)
    if tracker_runner is None:
        detector = YoloDetector(model_path, device=config.device)
        tracker_runner = TrackerRunner(
            detector,
            tracker=config.tracker,
            classes=config.classes,
            confidence=config.confidence,
            iou=config.iou,
            imgsz=config.imgsz,
        )

    run_config_path = _write_run_config(
        output_dir / "run_config.yaml",
        config=config,
        source=source,
        model_path=model_path,
    )
    line_counter = _build_line_counter(config)
    events_path = output_dir / "events.csv" if line_counter is not None else None
    evidence_dir = output_dir / "evidence" if line_counter is not None else None
    reset_peak_gpu_memory(config.device)
    latencies_ms: list[float] = []
    processed_frames = 0
    annotated_path = output_dir / "annotated.mp4"
    tracks_path = output_dir / "tracks.csv"

    with VideoSource(source) as video:
        metadata = video.metadata
        if metadata is None:
            raise PipelineError("Video metadata was unavailable after opening the source")
        with ExitStack() as stack:
            video_writer = stack.enter_context(
                AnnotatedVideoWriter(
                    annotated_path,
                    fps=metadata.fps,
                    width=metadata.width,
                    height=metadata.height,
                )
            )
            csv_writer = stack.enter_context(TrackCsvWriter(tracks_path))
            event_writer = (
                stack.enter_context(EventCsvWriter(events_path))
                if events_path is not None
                else None
            )
            for video_frame in video:
                frame_start = time.perf_counter()
                records = tracker_runner.process(
                    video_frame.image,
                    frame_id=video_frame.frame_id,
                    timestamp_ms=video_frame.timestamp_ms,
                )
                csv_writer.write(records)
                events = (
                    line_counter.process_frame(records, frame_id=video_frame.frame_id)
                    if line_counter is not None
                    else []
                )
                annotated = annotate_frame(
                    video_frame.image,
                    records,
                    counting_line=line_counter.line if line_counter is not None else None,
                    total_enter=line_counter.total_enter if line_counter is not None else 0,
                    total_exit=line_counter.total_exit if line_counter is not None else 0,
                )
                if events:
                    if event_writer is None or evidence_dir is None:
                        raise PipelineError("Counting events have no configured output writer")
                    persisted_events = []
                    for event in events:
                        filename = (
                            f"{event.event_id}_frame_{event.frame_id:06d}_"
                            f"track_{event.track_id}_{event.event_type}.jpg"
                        )
                        relative_evidence_path = Path("evidence") / filename
                        write_evidence_frame(evidence_dir / filename, annotated)
                        persisted_event = event.with_evidence_path(relative_evidence_path)
                        persisted_events.append(persisted_event)
                        logger.info(
                            "Counted %s: track=%s frame=%s evidence=%s",
                            persisted_event.event_type,
                            persisted_event.track_id,
                            persisted_event.frame_id,
                            relative_evidence_path,
                        )
                    event_writer.write(persisted_events)
                video_writer.write(annotated)
                processed_frames += 1
                latencies_ms.append((time.perf_counter() - frame_start) * 1000.0)
                if processed_frames % 100 == 0:
                    logger.info("Processed %s frames", processed_frames)
            if metadata.frame_count > 0 and processed_frames != metadata.frame_count:
                raise PipelineError(
                    f"Video decoding ended after {processed_frames} frames, "
                    f"but source metadata declared {metadata.frame_count}"
                )

    total_runtime_seconds = time.perf_counter() - total_start
    metrics = build_runtime_metrics(
        model=config.model,
        tracker=config.tracker,
        device=config.device,
        input_frames=metadata.frame_count,
        processed_frames=processed_frames,
        video_fps=metadata.fps,
        latencies_ms=latencies_ms,
        peak_gpu_memory_mb=read_peak_gpu_memory_mb(config.device),
        total_runtime_seconds=total_runtime_seconds,
    )
    metrics_path = write_json(output_dir / "runtime_metrics.json", metrics.as_dict())
    logger.info(
        "Completed run: frames=%s processing_fps=%.3f output=%s",
        processed_frames,
        metrics.processing_fps,
        output_dir,
    )
    if line_counter is not None:
        logger.info(
            "Directional counts: enter=%s exit=%s events=%s",
            line_counter.total_enter,
            line_counter.total_exit,
            events_path,
        )

    return PipelineResult(
        output_dir=output_dir,
        annotated_video=annotated_path,
        tracks_csv=tracks_path,
        run_config=run_config_path,
        events_csv=events_path,
        evidence_dir=evidence_dir,
        runtime_metrics=metrics_path,
        run_log=run_log,
        metrics=metrics,
        total_enter=line_counter.total_enter if line_counter is not None else 0,
        total_exit=line_counter.total_exit if line_counter is not None else 0,
    )
