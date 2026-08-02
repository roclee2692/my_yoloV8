"""Unit tests for loss-checked MOT image-sequence conversion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from people_flow.datasets.mot_converter import convert_mot_sequence_to_video
from people_flow.datasets.video_source import VideoSource
from people_flow.errors import MotConversionError


def _write_sequence(sequence_dir: Path, *, wrong_last_size: bool = False) -> Path:
    image_dir = sequence_dir / "img1"
    image_dir.mkdir(parents=True)
    (sequence_dir / "seqinfo.ini").write_text(
        "[Sequence]\n"
        f"name={sequence_dir.name}\n"
        "imDir=img1\n"
        "frameRate=7\n"
        "seqLength=3\n"
        "imWidth=32\n"
        "imHeight=24\n"
        "imExt=.jpg\n",
        encoding="utf-8",
    )
    for frame_id in range(1, 4):
        size = (20, 30) if wrong_last_size and frame_id == 3 else (24, 32)
        image = np.full((*size, 3), frame_id * 40, dtype=np.uint8)
        assert cv2.imwrite(str(image_dir / f"{frame_id:06d}.jpg"), image)
    return sequence_dir


def test_converter_preserves_frame_count_fps_and_dimensions(tmp_path: Path) -> None:
    """A successful summary must be backed by decoding every generated frame."""

    sequence = _write_sequence(tmp_path / "MOT17-99-SDP")
    output = tmp_path / "MOT17-99-SDP.mp4"

    result = convert_mot_sequence_to_video(sequence, output)

    assert result.input_frames == 3
    assert result.output_frames == 3
    assert result.frame_count_verified is True
    assert result.frame_rate == 7.0
    assert (result.width, result.height) == (32, 24)
    assert output.is_file()
    summary_path = tmp_path / "MOT17-99-SDP.summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["input_frames"] == payload["output_frames"] == 3
    assert payload["output_size_bytes"] == output.stat().st_size

    with VideoSource(output) as video:
        metadata = video.metadata
        decoded = list(video)
    assert metadata is not None
    assert metadata.fps == pytest.approx(7.0, abs=0.01)
    assert len(decoded) == 3


def test_converter_requires_explicit_overwrite(tmp_path: Path) -> None:
    """Existing artifacts must not be replaced without an explicit flag."""

    sequence = _write_sequence(tmp_path / "MOT17-99-SDP")
    output = tmp_path / "output.mp4"
    convert_mot_sequence_to_video(sequence, output)

    with pytest.raises(MotConversionError, match="--overwrite"):
        convert_mot_sequence_to_video(sequence, output)

    replacement = convert_mot_sequence_to_video(sequence, output, overwrite=True)
    assert replacement.output_frames == 3


def test_converter_rejects_frame_dimension_mismatch(tmp_path: Path) -> None:
    """A frame that disagrees with seqinfo.ini must abort and leave no partial MP4."""

    sequence = _write_sequence(tmp_path / "MOT17-99-SDP", wrong_last_size=True)
    output = tmp_path / "output.mp4"

    with pytest.raises(MotConversionError, match="has size 30x20"):
        convert_mot_sequence_to_video(sequence, output)

    assert not output.exists()
    assert not (tmp_path / "output.part.mp4").exists()
