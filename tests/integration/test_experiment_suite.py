"""Model-free integration test for the controlled Phase 8 experiment suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
import yaml

from people_flow.config import (
    CountingLineSettings,
    CountingSettings,
    RoiSettings,
    RunConfig,
    TrackerName,
)
from people_flow.datasets.video_source import Frame
from people_flow.errors import OutputError
from people_flow.evaluation.experiment_runner import (
    ExperimentSpec,
    ExperimentSuiteConfig,
    GroundTruthSettings,
    load_experiment_suite,
    run_experiment_suite,
)
from people_flow.pipeline import PipelineResult, run_pipeline
from people_flow.tracking.track_record import TrackRecord


class _MatrixTracker:
    def process(self, frame: Frame, *, frame_id: int, timestamp_ms: float) -> list[TrackRecord]:
        foot_y = {1: 5.0, 2: 14.0, 3: 22.0}[frame_id]
        return [
            TrackRecord.from_bbox(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                track_id=7,
                class_id=0,
                confidence=0.9,
                bbox=(9.0, foot_y - 5.0, 15.0, foot_y),
            )
        ]


def _create_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened()
    for value in (20, 40, 60):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


def _pipeline(config: RunConfig) -> PipelineResult:
    return run_pipeline(config, tracker_runner=_MatrixTracker())


def _create_mot_sequence(root: Path) -> Path:
    sequence = root / "MOT17-99-SDP"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "seqinfo.ini").write_text(
        """[Sequence]
name=MOT17-99-SDP
imDir=img1
frameRate=10
seqLength=3
imWidth=32
imHeight=24
imExt=.jpg
""",
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        """1,1,9,0,6,5,1,1,1
2,1,9,9,6,5,1,1,1
3,1,9,17,6,5,1,1,1
""",
        encoding="utf-8",
    )
    return sequence


@pytest.mark.slow
def test_experiment_suite_writes_three_comparable_runs_without_fake_gt(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    model = tmp_path / "fake.pt"
    _create_video(source)
    model.write_bytes(b"model-free-test")
    config = ExperimentSuiteConfig(
        schema_version=1,
        suite_id="test_suite",
        seed=42,
        source=source,
        output_root=tmp_path / "experiments",
        weights_dir=tmp_path / "weights",
        device="cpu",
        counting=CountingSettings(
            enabled=True,
            line=CountingLineSettings(p1=(0.0, 12.0), p2=(31.0, 12.0)),
            min_track_age=2,
            min_displacement_pixels=5.0,
            cooldown_frames=0,
            max_track_gap_frames=2,
        ),
        roi=RoiSettings(
            enabled=True,
            name="inside",
            points=((0.0, 10.0), (31.0, 10.0), (31.0, 18.0), (0.0, 18.0)),
            max_track_gap_frames=2,
        ),
        ground_truth=GroundTruthSettings(
            mot_sequence=None,
            unavailable_reason="Synthetic video has no Ground Truth.",
        ),
        experiments=tuple(
            ExperimentSpec(
                id=experiment_id,
                model=str(model),
                tracker=cast(TrackerName, tracker),
            )
            for experiment_id, tracker in (
                ("A", "bytetrack.yaml"),
                ("B", "bytetrack.yaml"),
                ("C", "botsort.yaml"),
            )
        ),
    )

    result = run_experiment_suite(config, base_dir=tmp_path, pipeline_runner=_pipeline)

    assert len(result.rows) == 3
    with result.comparison_csv.open(encoding="utf-8", newline="") as stream:
        comparison = list(csv.DictReader(stream))
    assert [row["experiment_id"] for row in comparison] == ["A", "B", "C"]
    assert {row["ground_truth_available"] for row in comparison} == {"False"}
    assert {row["id_switches"] for row in comparison} == {""}
    assert {row["track_observations"] for row in comparison} == {"3"}
    assert {row["unique_track_ids"] for row in comparison} == {"1"}

    required = {
        "config.yaml",
        "annotated.mp4",
        "tracks.csv",
        "events.csv",
        "occupancy.csv",
        "summary.json",
        "runtime_metrics.json",
        "evaluation.json",
        "run.log",
    }
    for experiment_id in ("A", "B", "C"):
        output = result.output_root / experiment_id
        assert required <= {path.name for path in output.iterdir()}
        effective = yaml.safe_load((output / "config.yaml").read_text(encoding="utf-8"))
        assert set(effective["software"]) == {
            "people_flow",
            "python",
            "opencv",
            "torch",
            "ultralytics",
        }
        evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        assert evaluation["ground_truth_available"] is False
        assert evaluation["id_switches"] is None
        assert evaluation["count_mae"] is None
        assert evaluation["occupancy_mae"] is None

    with pytest.raises(OutputError, match="Comparison output already exists"):
        run_experiment_suite(config, base_dir=tmp_path, pipeline_runner=_pipeline)


@pytest.mark.slow
def test_experiment_suite_computes_gt_metrics_when_mot_sequence_is_supplied(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.mp4"
    model = tmp_path / "fake.pt"
    _create_video(source)
    model.write_bytes(b"model-free-test")
    mot_sequence = _create_mot_sequence(tmp_path)
    config = ExperimentSuiteConfig(
        schema_version=1,
        suite_id="gt_suite",
        source=source,
        output_root=tmp_path / "gt_experiments",
        device="cpu",
        counting=CountingSettings(
            enabled=True,
            line=CountingLineSettings(p1=(0.0, 12.0), p2=(31.0, 12.0)),
            min_track_age=2,
            min_displacement_pixels=5.0,
            cooldown_frames=0,
            max_track_gap_frames=2,
        ),
        roi=RoiSettings(
            enabled=True,
            name="inside",
            points=((0.0, 10.0), (31.0, 10.0), (31.0, 18.0), (0.0, 18.0)),
            max_track_gap_frames=2,
        ),
        ground_truth=GroundTruthSettings(mot_sequence=mot_sequence),
        experiments=(ExperimentSpec(id="GT", model=str(model), tracker="bytetrack.yaml"),),
    )

    result = run_experiment_suite(config, base_dir=tmp_path, pipeline_runner=_pipeline)
    evaluation = json.loads(
        (result.output_root / "GT" / "evaluation.json").read_text(encoding="utf-8")
    )

    assert evaluation["ground_truth_available"] is True
    assert evaluation["id_switches"] == 0
    assert evaluation["enter_absolute_error"] == 0
    assert evaluation["exit_absolute_error"] == 0
    assert evaluation["total_count_absolute_error"] == 0
    assert evaluation["count_mae"] == 0.0
    assert evaluation["count_mape"] == 0.0
    assert evaluation["occupancy_mae"] == 0.0


def test_repository_baseline_config_declares_required_matrix() -> None:
    config = load_experiment_suite(Path("configs/experiments/baseline.yaml"))
    assert [(item.model, item.tracker) for item in config.experiments] == [
        ("yolov8n.pt", "bytetrack.yaml"),
        ("yolo26n.pt", "bytetrack.yaml"),
        ("yolo26n.pt", "botsort.yaml"),
    ]
    assert config.ground_truth.mot_sequence is None
    assert config.counting.enabled is True
    assert config.roi.enabled is True
