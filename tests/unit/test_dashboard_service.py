"""Tests for Dashboard input handling and core-pipeline delegation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

from people_flow.config import RunConfig
from people_flow.dashboard.service import (
    DashboardRunRequest,
    load_video_preview,
    map_display_point,
    persist_uploaded_mp4,
    run_dashboard_job,
    validate_run_id,
)
from people_flow.errors import DashboardError
from people_flow.pipeline import PipelineResult, ProgressCallback


def _create_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    writer.write(np.full((24, 32, 3), 50, dtype=np.uint8))
    writer.release()


def _request(tmp_path: Path) -> DashboardRunRequest:
    return DashboardRunRequest(
        source=tmp_path / "source.mp4",
        output_dir=tmp_path / "runs" / "demo",
        model="yolo26n.pt",
        tracker="bytetrack.yaml",
        device="cpu",
        confidence=0.25,
        line_points=((0.0, 12.0), (31.0, 12.0)),
        roi_points=((0.0, 0.0), (31.0, 0.0), (31.0, 23.0), (0.0, 23.0)),
    )


def test_upload_is_sanitized_content_addressed_and_mp4_only(tmp_path: Path) -> None:
    """Browser filenames cannot escape the upload folder or collide by content."""

    uploaded = persist_uploaded_mp4(
        "../../camera one.mp4",
        b"fake-mp4-content",
        upload_dir=tmp_path / "uploads",
    )

    assert uploaded.parent == (tmp_path / "uploads").resolve()
    assert uploaded.name.startswith("camera_one-")
    assert uploaded.suffix == ".mp4"
    assert persist_uploaded_mp4(
        "camera one.mp4",
        b"fake-mp4-content",
        upload_dir=tmp_path / "uploads",
    ) == uploaded
    with pytest.raises(DashboardError, match="must be MP4"):
        persist_uploaded_mp4("camera.avi", b"content", upload_dir=tmp_path)


def test_video_preview_and_coordinate_mapping_use_source_dimensions(tmp_path: Path) -> None:
    """Clicked preview points should map back to the original video pixels."""

    source = tmp_path / "source.mp4"
    _create_video(source)

    preview = load_video_preview(source)

    assert (preview.width, preview.height, preview.frame_count) == (32, 24, 1)
    assert preview.fps == pytest.approx(10.0, abs=0.1)
    assert preview.frame_rgb.shape == (24, 32, 3)
    assert map_display_point(
        16.0,
        12.0,
        display_size=(32, 24),
        source_size=(1280, 720),
    ) == (640.0, 360.0)


def test_dashboard_request_delegates_to_canonical_pipeline(tmp_path: Path) -> None:
    """The UI service must pass a canonical person-only RunConfig to one core runner."""

    request = _request(tmp_path)
    captured: list[RunConfig] = []
    sentinel = cast(PipelineResult, object())

    def fake_runner(
        config: RunConfig,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineResult:
        captured.append(config)
        assert progress_callback is None
        return sentinel

    result = run_dashboard_job(request, pipeline_runner=fake_runner)

    assert result is sentinel
    assert captured[0].classes == (0,)
    assert captured[0].counting.enabled is True
    assert captured[0].roi.enabled is True
    assert captured[0].tracker == "bytetrack.yaml"


@pytest.mark.parametrize("value", ("../escape", "", "has space"))
def test_validate_run_id_rejects_non_portable_paths(value: str) -> None:
    """Run IDs cannot become traversal paths or ambiguous folder names."""

    with pytest.raises(DashboardError, match="Run ID"):
        validate_run_id(value)
