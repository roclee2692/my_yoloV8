"""Phase 3 person detection and multi-object tracking pipeline."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import cv2
import yaml

from people_flow import __version__
from people_flow.assets import resolve_model_path
from people_flow.config import RunConfig, resolve_path
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
from people_flow.outputs.json_writer import write_json
from people_flow.outputs.video_writer import AnnotatedVideoWriter
from people_flow.tracking.track_record import TrackRecord
from people_flow.tracking.tracker_runner import TrackerRunner
from people_flow.visualization.annotator import annotate_frame

_OUTPUT_NAMES = (
    "annotated.mp4",
    "tracks.csv",
    "run_config.yaml",
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
    runtime_metrics: Path
    run_log: Path
    metrics: RuntimeMetrics


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> Path:
    """Create the output directory while preserving unrelated files."""

    resolved = output_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    existing = [resolved / name for name in _OUTPUT_NAMES if (resolved / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise OutputError(
            f"Output files already exist in {resolved}: {names}. Use --overwrite to replace them."
        )
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


def run_pipeline(
    config: RunConfig, *, tracker_runner: FrameTracker | None = None
) -> PipelineResult:
    """Run person-only tracking on a local video and write all Phase 3 artifacts."""

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
    reset_peak_gpu_memory(config.device)
    latencies_ms: list[float] = []
    processed_frames = 0
    annotated_path = output_dir / "annotated.mp4"
    tracks_path = output_dir / "tracks.csv"

    with VideoSource(source) as video:
        metadata = video.metadata
        if metadata is None:
            raise PipelineError("Video metadata was unavailable after opening the source")
        with (
            AnnotatedVideoWriter(
                annotated_path,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
            ) as video_writer,
            TrackCsvWriter(tracks_path) as csv_writer,
        ):
            for video_frame in video:
                frame_start = time.perf_counter()
                records = tracker_runner.process(
                    video_frame.image,
                    frame_id=video_frame.frame_id,
                    timestamp_ms=video_frame.timestamp_ms,
                )
                csv_writer.write(records)
                video_writer.write(annotate_frame(video_frame.image, records))
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
    return PipelineResult(
        output_dir=output_dir,
        annotated_video=annotated_path,
        tracks_csv=tracks_path,
        run_config=run_config_path,
        runtime_metrics=metrics_path,
        run_log=run_log,
        metrics=metrics,
    )
