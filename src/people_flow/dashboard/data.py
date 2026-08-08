"""Validated, source-backed reads for completed people-flow runs."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from people_flow.errors import DashboardError, ReportGenerationError
from people_flow.reporting.evidence import load_report_evidence

_REQUIRED_RUN_FILES = (
    "annotated.mp4",
    "tracks.csv",
    "run_config.yaml",
    "runtime_metrics.json",
    "run.log",
)


@dataclass(frozen=True, slots=True)
class DashboardRunData:
    """One completed run and its reconciled dashboard datasets."""

    output_dir: Path
    annotated_video: Path
    tracks: tuple[dict[str, str], ...]
    events: tuple[dict[str, str], ...]
    occupancy: tuple[dict[str, str], ...]
    dwell_times: tuple[dict[str, str], ...]
    runtime: dict[str, Any]
    summary: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    report_markdown: str | None
    report_metadata: dict[str, Any] | None

    @property
    def processed_frames(self) -> int:
        return _as_int(self.runtime, "processed_frames")

    @property
    def current_occupancy(self) -> int:
        if not self.occupancy:
            return 0
        return int(self.occupancy[-1]["occupancy"])

    @property
    def total_enter(self) -> int:
        return sum(row["event_type"] == "enter" for row in self.events)

    @property
    def total_exit(self) -> int:
        return sum(row["event_type"] == "exit" for row in self.events)

    @property
    def maximum_occupancy(self) -> int:
        if self.summary is not None:
            return _as_int(self.summary, "maximum_occupancy")
        return max((int(row["occupancy"]) for row in self.occupancy), default=0)

    @property
    def peak_time_ms(self) -> float | None:
        if self.summary is None or self.summary.get("peak_time") is None:
            return None
        return float(self.summary["peak_time"])

    @property
    def processing_fps(self) -> float:
        return _as_float(self.runtime, "processing_fps")

    @property
    def ground_truth_available(self) -> bool:
        return bool(self.evaluation and self.evaluation.get("ground_truth_available") is True)

    @property
    def ground_truth_message(self) -> str:
        if self.ground_truth_available:
            return "本次结果已关联 Ground Truth；请查看 evaluation.json 中的误差指标。"
        if self.evaluation and isinstance(self.evaluation.get("reason"), str):
            return str(self.evaluation["reason"])
        return "本次结果没有人工或数据集 Ground Truth 验证，仅为模型预测结果。"

    def flow_series(self) -> tuple[dict[str, int | float], ...]:
        """Return frame-level occupancy and cumulative directional counts."""

        events_by_frame: dict[int, list[str]] = {}
        for event in self.events:
            events_by_frame.setdefault(int(event["frame_id"]), []).append(event["event_type"])
        cumulative_enter = 0
        cumulative_exit = 0
        rows: list[dict[str, int | float]] = []
        for snapshot in self.occupancy:
            frame_id = int(snapshot["frame_id"])
            for event_type in events_by_frame.get(frame_id, ()):
                cumulative_enter += event_type == "enter"
                cumulative_exit += event_type == "exit"
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp_seconds": float(snapshot["timestamp_ms"]) / 1000.0,
                    "occupancy": int(snapshot["occupancy"]),
                    "cumulative_enter": cumulative_enter,
                    "cumulative_exit": cumulative_exit,
                }
            )
        return tuple(rows)

    def dwell_histogram(self) -> dict[str, int]:
        """Return fixed, human-readable dwell-time buckets."""

        buckets = {"<1s": 0, "1–3s": 0, "3–5s": 0, "5–10s": 0, "≥10s": 0}
        for row in self.dwell_times:
            seconds = float(row["total_dwell_seconds"])
            if seconds < 1.0:
                buckets["<1s"] += 1
            elif seconds < 3.0:
                buckets["1–3s"] += 1
            elif seconds < 5.0:
                buckets["3–5s"] += 1
            elif seconds < 10.0:
                buckets["5–10s"] += 1
            else:
                buckets["≥10s"] += 1
        return buckets


def discover_completed_runs(runs_dir: Path) -> tuple[Path, ...]:
    """Find completed runs by required persisted artifacts, newest first."""

    resolved = runs_dir.expanduser().resolve()
    if not resolved.is_dir():
        return ()
    candidates: list[Path] = []
    for metrics_path in resolved.rglob("runtime_metrics.json"):
        run_dir = metrics_path.parent
        if all((run_dir / name).is_file() for name in _REQUIRED_RUN_FILES):
            candidates.append(run_dir)
    return tuple(
        sorted(
            set(candidates),
            key=lambda path: (path.stat().st_mtime_ns, str(path)),
            reverse=True,
        )
    )


def load_run_dashboard(run_dir: Path) -> DashboardRunData:
    """Load and reconcile one completed run without invoking model inference."""

    resolved = run_dir.expanduser().resolve()
    missing = [name for name in _REQUIRED_RUN_FILES if not (resolved / name).is_file()]
    if missing:
        raise DashboardError(f"Run is missing required dashboard artifacts {missing}: {resolved}")
    runtime = _read_json_object(resolved / "runtime_metrics.json")
    tracks = _read_csv(resolved / "tracks.csv")
    events = _read_csv_optional(resolved / "events.csv")
    occupancy = _read_csv_optional(resolved / "occupancy.csv")
    dwell_times = _read_csv_optional(resolved / "dwell_times.csv")
    summary = _read_json_optional(resolved / "summary.json")
    evaluation = _read_json_optional(resolved / "evaluation.json")
    report_path = resolved / "report.md"
    report_metadata_path = resolved / "report_metadata.json"
    if report_path.is_file() != report_metadata_path.is_file():
        raise DashboardError("report.md and report_metadata.json must exist together")
    report_markdown = _read_text(report_path) if report_path.is_file() else None
    report_metadata = _read_json_optional(report_metadata_path)
    if report_metadata is not None:
        if report_metadata.get("raw_video_read") is not False:
            raise DashboardError("Report metadata must confirm raw_video_read is false")
        if evaluation is not None and report_metadata.get(
            "ground_truth_available"
        ) != evaluation.get("ground_truth_available"):
            raise DashboardError("Report and evaluation Ground Truth states do not match")
        try:
            current_fingerprint = load_report_evidence(resolved).fingerprint
        except ReportGenerationError as exc:
            raise DashboardError(f"Report evidence is invalid: {exc}") from exc
        if report_metadata.get("evidence_sha256") != current_fingerprint:
            raise DashboardError("Report evidence fingerprint does not match current run JSON")
    processed_frames = _as_int(runtime, "processed_frames")
    if occupancy and len(occupancy) != processed_frames:
        raise DashboardError(
            f"Occupancy row count does not match processed frames: "
            f"{len(occupancy)} != {processed_frames}"
        )
    if summary is not None:
        summary_frames = _as_int(summary, "processed_frames")
        if summary_frames != processed_frames:
            raise DashboardError(
                f"Summary frame count does not match runtime metrics: "
                f"{summary_frames} != {processed_frames}"
            )
    return DashboardRunData(
        output_dir=resolved,
        annotated_video=resolved / "annotated.mp4",
        tracks=tracks,
        events=events,
        occupancy=occupancy,
        dwell_times=dwell_times,
        runtime=runtime,
        summary=summary,
        evaluation=evaluation,
        report_markdown=report_markdown,
        report_metadata=report_metadata,
    )


def load_comparison_rows(path: Path) -> tuple[dict[str, str], ...]:
    """Load the Phase 8 comparison table for dashboard display and download."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return ()
    rows = _read_csv(resolved)
    required = {"experiment_id", "model", "tracker", "processing_fps"}
    if rows and not required <= set(rows[0]):
        raise DashboardError(f"Experiment comparison is missing required columns: {resolved}")
    return rows


def build_results_archive(data: DashboardRunData) -> bytes:
    """Build a portable ZIP from one run without following symlinks."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data.output_dir.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(data.output_dir)
            except ValueError as exc:
                raise DashboardError(f"Run artifact escapes output directory: {path}") from exc
            archive.write(path, arcname=str(relative).replace("\\", "/"))
    return output.getvalue()


def _read_csv_optional(path: Path) -> tuple[dict[str, str], ...]:
    return _read_csv(path) if path.is_file() else ()


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    except OSError as exc:
        raise DashboardError(f"Unable to read dashboard CSV: {path}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DashboardError(f"Unable to read dashboard text: {path}") from exc


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    return _read_json_object(path) if path.is_file() else None


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardError(f"Unable to read dashboard JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise DashboardError(f"Dashboard JSON must contain an object: {path}")
    return payload


def _as_int(payload: dict[str, Any], field: str) -> int:
    try:
        return int(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardError(f"Dashboard field {field!r} must be an integer") from exc


def _as_float(payload: dict[str, Any], field: str) -> float:
    try:
        return float(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardError(f"Dashboard field {field!r} must be numeric") from exc
