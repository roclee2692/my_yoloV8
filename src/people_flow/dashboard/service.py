"""Thin Dashboard services that delegate processing to the core pipeline."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from people_flow.config import (
    CountingLineSettings,
    CountingSettings,
    RoiSettings,
    RunConfig,
    TrackerName,
)
from people_flow.counting.geometry import DirectedLine, PolygonRegion
from people_flow.errors import DashboardError
from people_flow.pipeline import PipelineResult, ProgressCallback, run_pipeline

Point = tuple[float, float]
PipelineRunner = Callable[..., PipelineResult]
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


@dataclass(frozen=True, slots=True)
class VideoPreview:
    """First-frame image and source metadata for geometry selection."""

    frame_rgb: NDArray[np.uint8]
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True, slots=True)
class DashboardRunRequest:
    """Validated Dashboard selections for one core pipeline invocation."""

    source: Path
    output_dir: Path
    model: str
    tracker: TrackerName
    device: str
    confidence: float
    line_points: tuple[Point, ...]
    roi_points: tuple[Point, ...]
    overwrite: bool = False
    weights_dir: Path = Path("weights")
    iou: float = 0.7
    imgsz: int = 640

    def to_run_config(self) -> RunConfig:
        """Build the canonical core RunConfig; geometry validation remains shared."""

        if len(self.line_points) != 2:
            raise DashboardError("Counting line requires exactly two endpoints")
        if len(self.roi_points) < 3:
            raise DashboardError("ROI requires at least three vertices")
        line = DirectedLine(self.line_points[0], self.line_points[1])
        roi = PolygonRegion("entrance_area", self.roi_points)
        return RunConfig(
            source=self.source,
            model=self.model,
            tracker=self.tracker,
            classes=(0,),
            output_dir=self.output_dir,
            weights_dir=self.weights_dir,
            device=self.device,
            confidence=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            overwrite=self.overwrite,
            counting=CountingSettings(
                enabled=True,
                line=CountingLineSettings(
                    p1=line.p1,
                    p2=line.p2,
                    enter_side="positive",
                ),
                min_track_age=5,
                min_displacement_pixels=15.0,
                cooldown_frames=30,
                max_track_gap_frames=30,
            ),
            roi=RoiSettings(
                enabled=True,
                name=roi.name,
                points=roi.points,
                max_track_gap_frames=30,
            ),
        )


def persist_uploaded_mp4(filename: str, content: bytes, *, upload_dir: Path) -> Path:
    """Persist an uploaded MP4 under a sanitized content-addressed filename."""

    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() != ".mp4":
        raise DashboardError("Dashboard uploads must be MP4 files")
    if not content:
        raise DashboardError("Uploaded MP4 is empty")
    digest = hashlib.sha256(content).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(safe_name).stem).strip("._") or "upload"
    resolved_dir = upload_dir.expanduser().resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    destination = resolved_dir / f"{stem}-{digest}.mp4"
    if destination.exists():
        return destination
    partial = destination.with_suffix(".mp4.part")
    try:
        partial.write_bytes(content)
        partial.replace(destination)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise DashboardError(f"Unable to persist uploaded video: {destination}") from exc
    return destination


def load_video_preview(path: Path) -> VideoPreview:
    """Decode the first source frame and validated video metadata."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DashboardError(f"Video source does not exist: {resolved}")
    capture = cv2.VideoCapture(str(resolved))
    try:
        if not capture.isOpened():
            raise DashboardError(f"Unable to open video source: {resolved}")
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            raise DashboardError(f"Unable to decode first video frame: {resolved}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise DashboardError(f"Video metadata is invalid: {resolved}")
    frame_rgb = cast(NDArray[np.uint8], cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return VideoPreview(frame_rgb, width, height, fps, frame_count)


def map_display_point(
    x: float,
    y: float,
    *,
    display_size: tuple[int, int],
    source_size: tuple[int, int],
) -> Point:
    """Map a clicked preview coordinate back into source-video pixels."""

    display_width, display_height = display_size
    source_width, source_height = source_size
    if min(display_width, display_height, source_width, source_height) <= 0:
        raise DashboardError("Display and source dimensions must be positive")
    mapped_x = min(max(float(x) * source_width / display_width, 0.0), source_width - 1.0)
    mapped_y = min(max(float(y) * source_height / display_height, 0.0), source_height - 1.0)
    return (mapped_x, mapped_y)


def draw_geometry_preview(
    frame_rgb: NDArray[np.uint8],
    *,
    line_points: Sequence[Point],
    roi_points: Sequence[Point],
) -> NDArray[np.uint8]:
    """Draw UI-only point/segment feedback without implementing counting logic."""

    preview = frame_rgb.copy()
    _draw_points(preview, line_points, color=(255, 85, 85))
    _draw_points(preview, roi_points, color=(85, 220, 160))
    if len(line_points) == 2:
        p1, p2 = (_int_point(point) for point in line_points)
        cv2.line(preview, p1, p2, (255, 85, 85), 3, cv2.LINE_AA)
    if len(roi_points) >= 2:
        points = np.asarray([_int_point(point) for point in roi_points], dtype=np.int32)
        cv2.polylines(
            preview,
            [points],
            isClosed=len(roi_points) >= 3,
            color=(85, 220, 160),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    return preview


def validate_run_id(value: str) -> str:
    """Validate a portable run ID before joining it to the runs directory."""

    normalized = value.strip()
    if not _SAFE_RUN_ID.fullmatch(normalized):
        raise DashboardError(
            "Run ID must start with a letter or digit and use at most 80 portable characters"
        )
    return normalized


def resolve_dashboard_output_dir(requested: Path, *, overwrite: bool) -> Path:
    """Return a non-destructive output directory for a Dashboard run.

    When overwrite is disabled, an existing requested directory is preserved and
    the next available numbered suffix is selected.
    """

    resolved = requested.expanduser().resolve()
    if overwrite or not resolved.exists():
        return resolved
    for index in range(2, 10_000):
        candidate = resolved.with_name(f"{resolved.name}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise DashboardError(f"Unable to allocate a new output directory for: {resolved}")


def run_dashboard_job(
    request: DashboardRunRequest,
    *,
    progress_callback: ProgressCallback | None = None,
    pipeline_runner: PipelineRunner = run_pipeline,
) -> PipelineResult:
    """Invoke the canonical pipeline with Dashboard-selected configuration."""

    return pipeline_runner(request.to_run_config(), progress_callback=progress_callback)


def _draw_points(
    image: NDArray[np.uint8],
    points: Sequence[Point],
    *,
    color: tuple[int, int, int],
) -> None:
    for index, point in enumerate(points, start=1):
        position = _int_point(point)
        cv2.circle(image, position, 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            str(index),
            (position[0] + 9, position[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def _int_point(point: Point) -> tuple[int, int]:
    return (int(round(point[0])), int(round(point[1])))
