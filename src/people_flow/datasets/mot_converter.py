"""Loss-checked conversion of MOT image sequences to ordinary MP4 video."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from people_flow.datasets.mot_reader import MotSequence
from people_flow.datasets.video_source import VideoSource
from people_flow.errors import MotConversionError, VideoSourceError
from people_flow.outputs.json_writer import write_json

Frame = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class MotVideoConversionSummary:
    """Auditable result of one verified MOT-to-MP4 conversion."""

    sequence_name: str
    sequence_path: str
    output_path: str
    input_frames: int
    output_frames: int
    frame_rate: float
    width: int
    height: int
    image_extension: str
    codec: str
    output_size_bytes: int
    frame_count_verified: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return asdict(self)


def convert_mot_sequence_to_video(
    sequence_dir: Path,
    output_path: Path,
    *,
    summary_path: Path | None = None,
    codec: str = "mp4v",
    overwrite: bool = False,
) -> MotVideoConversionSummary:
    """Convert every declared MOT frame, then decode the MP4 to prove frame parity."""

    sequence = MotSequence(sequence_dir)
    frame_paths = sequence.image_paths()
    output = output_path.expanduser().resolve()
    summary = (
        summary_path.expanduser().resolve()
        if summary_path is not None
        else output.with_suffix(".summary.json")
    )
    _validate_output_paths(output, summary, overwrite)
    if len(codec) != 4:
        raise MotConversionError(f"Video codec must contain exactly four characters: {codec!r}")

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MotConversionError(
            f"Unable to create video output directory: {output.parent}"
        ) from exc
    temporary_output = output.with_name(f"{output.stem}.part{output.suffix}")
    try:
        if temporary_output.exists():
            temporary_output.unlink()
    except OSError as exc:
        raise MotConversionError(
            f"Unable to remove stale temporary video: {temporary_output}"
        ) from exc

    writer = cv2.VideoWriter(
        str(temporary_output),
        cv2.VideoWriter.fourcc(*codec),
        sequence.info.frame_rate,
        (sequence.info.image_width, sequence.info.image_height),
    )
    if not writer.isOpened():
        writer.release()
        temporary_output.unlink(missing_ok=True)
        raise MotConversionError(
            f"OpenCV could not open the MP4 writer for codec {codec!r}: {output}"
        )

    written_frames = 0
    conversion_completed = False
    try:
        for frame_id, frame_path in enumerate(frame_paths, start=1):
            frame = cast(Frame | None, cv2.imread(str(frame_path), cv2.IMREAD_COLOR))
            if frame is None:
                raise MotConversionError(
                    f"OpenCV could not decode MOT frame {frame_id}: {frame_path}"
                )
            height, width = frame.shape[:2]
            if (width, height) != (sequence.info.image_width, sequence.info.image_height):
                raise MotConversionError(
                    f"MOT frame {frame_id} has size {width}x{height}; "
                    "seqinfo.ini declares "
                    f"{sequence.info.image_width}x{sequence.info.image_height}: "
                    f"{frame_path}"
                )
            writer.write(frame)
            written_frames += 1
        conversion_completed = True
    finally:
        writer.release()
        if not conversion_completed:
            temporary_output.unlink(missing_ok=True)

    if written_frames != sequence.info.sequence_length:
        temporary_output.unlink(missing_ok=True)
        raise MotConversionError(
            f"Conversion wrote {written_frames} of {sequence.info.sequence_length} declared frames"
        )

    try:
        decoded_frames = _verify_video(temporary_output, sequence)
    except (VideoSourceError, MotConversionError):
        temporary_output.unlink(missing_ok=True)
        raise

    try:
        temporary_output.replace(output)
    except OSError as exc:
        temporary_output.unlink(missing_ok=True)
        raise MotConversionError(f"Unable to finalize converted video: {output}") from exc

    result = MotVideoConversionSummary(
        sequence_name=sequence.info.name,
        sequence_path=str(sequence.sequence_dir),
        output_path=str(output),
        input_frames=len(frame_paths),
        output_frames=decoded_frames,
        frame_rate=sequence.info.frame_rate,
        width=sequence.info.image_width,
        height=sequence.info.image_height,
        image_extension=sequence.info.image_extension,
        codec=codec,
        output_size_bytes=output.stat().st_size,
        frame_count_verified=decoded_frames == len(frame_paths),
    )
    write_json(summary, result.as_dict())
    return result


def _validate_output_paths(output: Path, summary: Path, overwrite: bool) -> None:
    if output.suffix.lower() != ".mp4":
        raise MotConversionError(f"MOT conversion output must use the .mp4 extension: {output}")
    if output == summary:
        raise MotConversionError("Video output and JSON summary paths must be different")
    existing = [path for path in (output, summary) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise MotConversionError(f"Output already exists; pass --overwrite to replace it: {joined}")


def _verify_video(video_path: Path, sequence: MotSequence) -> int:
    decoded_frames = 0
    try:
        with VideoSource(video_path) as source:
            metadata = source.metadata
            if metadata is None:
                raise MotConversionError(f"Converted video metadata is unavailable: {video_path}")
            expected_size = (sequence.info.image_width, sequence.info.image_height)
            actual_size = (metadata.width, metadata.height)
            if actual_size != expected_size:
                raise MotConversionError(
                    f"Converted video size is {actual_size}, expected {expected_size}: {video_path}"
                )
            if abs(metadata.fps - sequence.info.frame_rate) > 0.01:
                raise MotConversionError(
                    f"Converted video FPS is {metadata.fps}, expected {sequence.info.frame_rate}: "
                    f"{video_path}"
                )
            for _frame in source:
                decoded_frames += 1
    except VideoSourceError as exc:
        raise MotConversionError(f"Unable to verify converted video: {video_path}") from exc

    if decoded_frames != sequence.info.sequence_length:
        raise MotConversionError(
            f"Converted video decodes to {decoded_frames} frames, "
            f"expected {sequence.info.sequence_length}: {video_path}"
        )
    return decoded_frames
