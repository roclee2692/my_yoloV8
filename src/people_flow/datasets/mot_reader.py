"""Strict reader for MOTChallenge image sequences and MOT16/17 ground truth."""

from __future__ import annotations

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from people_flow.errors import MotFormatError

_MOT17_BASE_SEQUENCE_IDS = {
    "train": ("02", "04", "05", "09", "10", "11", "13"),
    "test": ("01", "03", "06", "07", "08", "12", "14"),
}
_MOT17_DETECTOR_VARIANTS = ("DPM", "FRCNN", "SDP")


@dataclass(frozen=True, slots=True)
class MotSequenceInfo:
    """Metadata declared by one MOT ``seqinfo.ini`` file."""

    sequence_dir: Path
    name: str
    image_dir: Path
    frame_rate: float
    sequence_length: int
    image_width: int
    image_height: int
    image_extension: str


@dataclass(frozen=True, slots=True)
class MotGroundTruthRecord:
    """One nine-column MOT16/17 ground-truth observation."""

    frame_id: int
    track_id: int
    bbox_left: float
    bbox_top: float
    bbox_width: float
    bbox_height: float
    confidence: float
    class_id: int
    visibility: float

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Return the bounding box as ``(left, top, right, bottom)``."""

        return (
            self.bbox_left,
            self.bbox_top,
            self.bbox_left + self.bbox_width,
            self.bbox_top + self.bbox_height,
        )

    @property
    def foot_point(self) -> tuple[float, float]:
        """Return the bottom-center point used by later counting phases."""

        return (
            self.bbox_left + self.bbox_width / 2.0,
            self.bbox_top + self.bbox_height,
        )


class MotSequence:
    """Read and validate a MOTChallenge sequence without changing source files."""

    _REQUIRED_KEYS = (
        "name",
        "imDir",
        "frameRate",
        "seqLength",
        "imWidth",
        "imHeight",
        "imExt",
    )

    def __init__(self, sequence_dir: Path) -> None:
        self.sequence_dir = sequence_dir.expanduser().resolve()
        self.info = self._read_sequence_info()

    def _read_sequence_info(self) -> MotSequenceInfo:
        if not self.sequence_dir.is_dir():
            raise MotFormatError(f"MOT sequence directory does not exist: {self.sequence_dir}")

        seqinfo_path = self.sequence_dir / "seqinfo.ini"
        if not seqinfo_path.is_file():
            raise MotFormatError(f"MOT sequence is missing seqinfo.ini: {self.sequence_dir}")

        parser = configparser.ConfigParser(interpolation=None)
        try:
            with seqinfo_path.open(encoding="utf-8-sig") as stream:
                parser.read_file(stream)
        except (OSError, configparser.Error) as exc:
            raise MotFormatError(f"Unable to parse MOT metadata: {seqinfo_path}") from exc

        if not parser.has_section("Sequence"):
            raise MotFormatError(f"MOT metadata has no [Sequence] section: {seqinfo_path}")
        section = parser["Sequence"]
        missing = [key for key in self._REQUIRED_KEYS if key not in section]
        if missing:
            joined = ", ".join(missing)
            raise MotFormatError(f"MOT metadata is missing keys ({joined}): {seqinfo_path}")

        name = section["name"].strip()
        image_directory_name = section["imDir"].strip()
        image_extension = section["imExt"].strip()
        if not name or not image_directory_name:
            raise MotFormatError(f"MOT metadata contains an empty name or imDir: {seqinfo_path}")
        if not image_extension.startswith("."):
            raise MotFormatError(
                f"MOT imExt must start with a dot, got {image_extension!r}: {seqinfo_path}"
            )

        frame_rate = _parse_positive_float(section["frameRate"], "frameRate", seqinfo_path)
        sequence_length = _parse_positive_int(section["seqLength"], "seqLength", seqinfo_path)
        image_width = _parse_positive_int(section["imWidth"], "imWidth", seqinfo_path)
        image_height = _parse_positive_int(section["imHeight"], "imHeight", seqinfo_path)
        image_dir = (self.sequence_dir / image_directory_name).resolve()

        try:
            image_dir.relative_to(self.sequence_dir)
        except ValueError as exc:
            raise MotFormatError(
                f"MOT imDir escapes the sequence directory: {image_directory_name!r}"
            ) from exc
        if not image_dir.is_dir():
            raise MotFormatError(f"MOT image directory does not exist: {image_dir}")

        return MotSequenceInfo(
            sequence_dir=self.sequence_dir,
            name=name,
            image_dir=image_dir,
            frame_rate=frame_rate,
            sequence_length=sequence_length,
            image_width=image_width,
            image_height=image_height,
            image_extension=image_extension,
        )

    def image_paths(self) -> tuple[Path, ...]:
        """Return numerically ordered frames after checking for gaps and extras."""

        extension = self.info.image_extension.lower()
        indexed_paths: dict[int, Path] = {}
        for path in self.info.image_dir.iterdir():
            if not path.is_file() or path.suffix.lower() != extension:
                continue
            if not path.stem.isdecimal():
                raise MotFormatError(f"MOT frame name is not numeric: {path}")
            frame_id = int(path.stem)
            expected_name = f"{frame_id:06d}{self.info.image_extension}"
            if path.name != expected_name:
                raise MotFormatError(
                    "MOT17 frame must use a six-digit name; "
                    f"expected {expected_name}, got {path.name}"
                )
            if frame_id in indexed_paths:
                raise MotFormatError(
                    f"MOT sequence has duplicate numeric frame ID {frame_id}: {path}"
                )
            indexed_paths[frame_id] = path.resolve()

        expected_ids = set(range(1, self.info.sequence_length + 1))
        actual_ids = set(indexed_paths)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extras = sorted(actual_ids - expected_ids)
            raise MotFormatError(
                "MOT frame set does not match seqLength "
                f"for {self.info.name}: missing={_short_ids(missing)}, "
                f"extras={_short_ids(extras)}"
            )
        return tuple(indexed_paths[frame_id] for frame_id in sorted(indexed_paths))

    @property
    def ground_truth_path(self) -> Path:
        """Return the conventional MOT ground-truth file path."""

        return self.sequence_dir / "gt" / "gt.txt"

    def read_ground_truth(self, *, required: bool = True) -> tuple[MotGroundTruthRecord, ...]:
        """Read all nine MOT16/17 GT columns without silently filtering records."""

        gt_path = self.ground_truth_path
        if not gt_path.is_file():
            if required:
                raise MotFormatError(f"MOT sequence is missing ground truth: {gt_path}")
            return ()

        records: list[MotGroundTruthRecord] = []
        try:
            lines = gt_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            raise MotFormatError(f"Unable to read MOT ground truth: {gt_path}") from exc

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 9:
                raise MotFormatError(
                    f"Expected 9 GT columns at {gt_path}:{line_number}, got {len(fields)}"
                )
            try:
                record = MotGroundTruthRecord(
                    frame_id=_parse_integral_number(fields[0], "frame"),
                    track_id=_parse_integral_number(fields[1], "track_id"),
                    bbox_left=_parse_finite_float(fields[2], "bbox_left"),
                    bbox_top=_parse_finite_float(fields[3], "bbox_top"),
                    bbox_width=_parse_finite_float(fields[4], "bbox_width"),
                    bbox_height=_parse_finite_float(fields[5], "bbox_height"),
                    confidence=_parse_finite_float(fields[6], "confidence"),
                    class_id=_parse_integral_number(fields[7], "class"),
                    visibility=_parse_finite_float(fields[8], "visibility"),
                )
            except ValueError as exc:
                raise MotFormatError(f"Invalid GT value at {gt_path}:{line_number}: {exc}") from exc
            self._validate_ground_truth_record(record, gt_path, line_number)
            records.append(record)
        return tuple(records)

    def _validate_ground_truth_record(
        self,
        record: MotGroundTruthRecord,
        gt_path: Path,
        line_number: int,
    ) -> None:
        location = f"{gt_path}:{line_number}"
        if not 1 <= record.frame_id <= self.info.sequence_length:
            raise MotFormatError(
                f"GT frame {record.frame_id} is outside 1..{self.info.sequence_length}: {location}"
            )
        if record.track_id <= 0:
            raise MotFormatError(f"GT track_id must be positive: {location}")
        if record.bbox_width <= 0 or record.bbox_height <= 0:
            raise MotFormatError(f"GT bounding-box width and height must be positive: {location}")
        if record.class_id <= 0:
            raise MotFormatError(f"GT class must be positive: {location}")
        if not 0.0 <= record.visibility <= 1.0:
            raise MotFormatError(f"GT visibility must be between 0 and 1: {location}")


def _parse_positive_int(value: str, field: str, path: Path) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise MotFormatError(f"MOT {field} must be an integer in {path}: {value!r}") from exc
    if parsed <= 0:
        raise MotFormatError(f"MOT {field} must be positive in {path}: {parsed}")
    return parsed


def _parse_positive_float(value: str, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise MotFormatError(f"MOT {field} must be numeric in {path}: {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise MotFormatError(f"MOT {field} must be a positive finite value in {path}: {value!r}")
    return parsed


def _parse_finite_float(value: str, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _parse_integral_number(value: str, field: str) -> int:
    parsed = _parse_finite_float(value, field)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be integral, got {value!r}")
    return int(parsed)


def _short_ids(ids: list[int]) -> str:
    if not ids:
        return "[]"
    visible = ", ".join(str(frame_id) for frame_id in ids[:10])
    suffix = ", ..." if len(ids) > 10 else ""
    return f"[{visible}{suffix}]"


def mot17_expected_sequence_names(split: str) -> tuple[str, ...]:
    """Return the 21 official MOT17 sequence-directory names for a split."""

    if split not in _MOT17_BASE_SEQUENCE_IDS:
        raise MotFormatError(f"MOT17 split must be 'train' or 'test', got {split!r}")
    return tuple(
        f"MOT17-{sequence_id}-{detector}"
        for sequence_id in _MOT17_BASE_SEQUENCE_IDS[split]
        for detector in _MOT17_DETECTOR_VARIANTS
    )


def validate_mot17_split(
    root: Path,
    split: str,
    *,
    allow_subset: bool = False,
) -> dict[str, Any]:
    """Validate each local MOT17 sequence and return measured split totals."""

    expected_names = frozenset(mot17_expected_sequence_names(split))
    resolved_root = root.expanduser().resolve()
    split_dir = resolved_root / split
    if not split_dir.is_dir():
        raise MotFormatError(
            f"MOT17 {split} directory does not exist: {split_dir}. "
            "Download and extract MOT17 according to data/README.md."
        )
    sequence_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir() and path.name.startswith("MOT17-")
    )
    if not sequence_dirs:
        raise MotFormatError(f"No MOT17 sequence directories found in: {split_dir}")
    actual_names = frozenset(path.name for path in sequence_dirs)
    official_split_complete = actual_names == expected_names
    if not allow_subset and not official_split_complete:
        missing = sorted(expected_names - actual_names)
        extras = sorted(actual_names - expected_names)
        raise MotFormatError(
            f"MOT17 {split} official split is incomplete: "
            f"missing={missing}, extras={extras}. "
            "Use --allow-subset only when intentionally validating selected sequences."
        )

    summaries: list[dict[str, Any]] = []
    total_frames = 0
    total_ground_truth_records = 0
    for sequence_dir in sequence_dirs:
        sequence = MotSequence(sequence_dir)
        frame_paths = sequence.image_paths()
        ground_truth = sequence.read_ground_truth(required=split == "train")
        total_frames += len(frame_paths)
        total_ground_truth_records += len(ground_truth)
        summaries.append(
            {
                "name": sequence.info.name,
                "path": str(sequence.sequence_dir),
                "frames": len(frame_paths),
                "frame_rate": sequence.info.frame_rate,
                "width": sequence.info.image_width,
                "height": sequence.info.image_height,
                "image_extension": sequence.info.image_extension,
                "ground_truth_records": len(ground_truth) if split == "train" else None,
            }
        )

    return {
        "dataset": "MOT17",
        "root": str(resolved_root),
        "split": split,
        "validation_mode": "subset" if allow_subset else "official_split",
        "expected_sequence_count": len(expected_names),
        "sequence_count": len(summaries),
        "total_frames": total_frames,
        "total_ground_truth_records": (total_ground_truth_records if split == "train" else None),
        "content_validated": True,
        "official_split_complete": official_split_complete,
        "complete": official_split_complete,
        "sequences": summaries,
    }
