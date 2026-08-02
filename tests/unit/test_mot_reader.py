"""Unit tests for strict MOT17 metadata, frame, and GT parsing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from people_flow.datasets.mot_ground_truth import load_mot_ground_truth
from people_flow.datasets.mot_reader import (
    MotSequence,
    mot17_expected_sequence_names,
    validate_mot17_split,
)
from people_flow.errors import MotFormatError


def _write_sequence(
    sequence_dir: Path,
    *,
    frame_ids: tuple[int, ...] = (3, 1, 2),
    include_ground_truth: bool = True,
) -> Path:
    """Create a minimal MOT17-compatible sequence with deliberately unordered writes."""

    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)
    (sequence_dir / "seqinfo.ini").write_text(
        "\n".join(
            (
                "[Sequence]",
                f"name={sequence_dir.name}",
                "imDir=img1",
                "frameRate=7",
                "seqLength=3",
                "imWidth=32",
                "imHeight=24",
                "imExt=.jpg",
                "",
            )
        ),
        encoding="utf-8",
    )
    for frame_id in frame_ids:
        image = np.full((24, 32, 3), frame_id * 30, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), image)
    if include_ground_truth:
        ground_truth_dir = sequence_dir / "gt"
        ground_truth_dir.mkdir()
        (ground_truth_dir / "gt.txt").write_text(
            "1,7,2.5,3.5,10,12,1,1,0.75\n2,7,3.5,3.5,10,12,1,1,0.50\n2,9,15,4,5,8,0,8,1.0\n",
            encoding="utf-8",
        )
    return sequence_dir


def test_sequence_reads_metadata_and_sorts_frames_numerically(tmp_path: Path) -> None:
    """Frame order must follow numeric IDs rather than directory iteration order."""

    sequence = MotSequence(_write_sequence(tmp_path / "MOT17-99-SDP"))

    assert sequence.info.name == "MOT17-99-SDP"
    assert sequence.info.frame_rate == 7.0
    assert sequence.info.sequence_length == 3
    assert (sequence.info.image_width, sequence.info.image_height) == (32, 24)
    assert [path.name for path in sequence.image_paths()] == [
        "000001.jpg",
        "000002.jpg",
        "000003.jpg",
    ]


def test_sequence_reads_all_nine_ground_truth_columns(tmp_path: Path) -> None:
    """GT parsing must preserve flags, classes, visibility, boxes, and IDs."""

    sequence_dir = _write_sequence(tmp_path / "MOT17-99-SDP")
    sequence = MotSequence(sequence_dir)
    records = sequence.read_ground_truth()

    assert len(records) == 3
    first = records[0]
    assert first.frame_id == 1
    assert first.track_id == 7
    assert first.bbox_xyxy == (2.5, 3.5, 12.5, 15.5)
    assert first.foot_point == (7.5, 15.5)
    assert first.confidence == 1.0
    assert first.class_id == 1
    assert first.visibility == 0.75

    ground_truth = load_mot_ground_truth(sequence_dir)
    assert ground_truth.track_ids == frozenset({7, 9})
    assert len(ground_truth.for_frame(2)) == 2
    assert ground_truth.for_frame(3) == ()


def test_sequence_rejects_a_missing_declared_frame(tmp_path: Path) -> None:
    """A gap in 1..seqLength must fail instead of producing a shorter video."""

    sequence = MotSequence(_write_sequence(tmp_path / "MOT17-99-SDP", frame_ids=(1, 3)))

    with pytest.raises(MotFormatError, match=r"missing=\[2\]"):
        sequence.image_paths()


def test_sequence_rejects_malformed_ground_truth(tmp_path: Path) -> None:
    """MOT17 GT rows require exactly nine fields."""

    sequence_dir = _write_sequence(tmp_path / "MOT17-99-SDP")
    (sequence_dir / "gt" / "gt.txt").write_text("1,7,2,3,4,5,1,1\n", encoding="utf-8")

    with pytest.raises(MotFormatError, match="Expected 9 GT columns"):
        MotSequence(sequence_dir).read_ground_truth()


def test_test_split_allows_absent_ground_truth(tmp_path: Path) -> None:
    """Official MOT17 test sequences have no public GT and should still validate."""

    sequence = MotSequence(_write_sequence(tmp_path / "MOT17-99-SDP", include_ground_truth=False))

    assert sequence.read_ground_truth(required=False) == ()
    with pytest.raises(MotFormatError, match="missing ground truth"):
        sequence.read_ground_truth(required=True)


def test_prepare_summary_marks_an_intentional_subset_incomplete(tmp_path: Path) -> None:
    """Subset validation must not claim that the official 21-sequence split is complete."""

    root = tmp_path / "MOT17"
    _write_sequence(root / "train" / "MOT17-99-SDP")

    summary = validate_mot17_split(root, "train", allow_subset=True)

    assert summary["dataset"] == "MOT17"
    assert summary["validation_mode"] == "subset"
    assert summary["content_validated"] is True
    assert summary["official_split_complete"] is False
    assert summary["complete"] is False
    assert summary["sequence_count"] == 1
    assert summary["total_frames"] == 3
    assert summary["total_ground_truth_records"] == 3


def test_prepare_rejects_an_unintentional_partial_official_split(tmp_path: Path) -> None:
    """Default validation should detect missing official sequence directories."""

    root = tmp_path / "MOT17"
    _write_sequence(root / "train" / "MOT17-02-DPM")

    with pytest.raises(MotFormatError, match="official split is incomplete"):
        validate_mot17_split(root, "train")


def test_prepare_accepts_all_official_training_sequence_names(tmp_path: Path) -> None:
    """The declared split is complete only when all 21 official variants validate."""

    root = tmp_path / "MOT17"
    for sequence_name in mot17_expected_sequence_names("train"):
        _write_sequence(root / "train" / sequence_name)

    summary = validate_mot17_split(root, "train")

    assert summary["expected_sequence_count"] == 21
    assert summary["sequence_count"] == 21
    assert summary["official_split_complete"] is True
    assert summary["complete"] is True
