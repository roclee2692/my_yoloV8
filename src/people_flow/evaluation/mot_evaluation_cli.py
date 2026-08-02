"""Evaluate predicted Track CSV against one MOT sequence Ground Truth."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from people_flow.config import CountingLineSettings, CountingSettings, RoiSettings, SideName
from people_flow.datasets.mot_ground_truth import GroundTruthPolicy, load_mot_ground_truth
from people_flow.errors import EvaluationError, PeopleFlowError
from people_flow.evaluation.trajectory_evaluator import (
    MotPredictionEvaluation,
    TrajectoryAnalysis,
    evaluate_mot_prediction,
)
from people_flow.evaluation.trajectory_io import read_track_records_csv
from people_flow.logging import configure_logging
from people_flow.outputs.event_writer import EventCsvWriter
from people_flow.outputs.json_writer import write_json
from people_flow.outputs.roi_writer import OccupancyCsvWriter, write_dwell_times_csv

_OUTPUT_NAMES = (
    "ground_truth_events.csv",
    "predicted_events.csv",
    "ground_truth_occupancy.csv",
    "predicted_occupancy.csv",
    "ground_truth_dwell_times.csv",
    "predicted_dwell_times.csv",
    "evaluation.json",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Ground Truth evaluation CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-sequence", required=True, type=Path)
    parser.add_argument("--predicted-tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--counting-line", required=True, nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--enter-side", choices=("positive", "negative"), default="positive")
    parser.add_argument("--min-track-age", type=int, default=5)
    parser.add_argument("--min-displacement-pixels", type=float, default=15.0)
    parser.add_argument("--cooldown-frames", type=int, default=30)
    parser.add_argument("--max-track-gap-frames", type=int, default=30)
    parser.add_argument("--roi-name", default="entrance_area")
    parser.add_argument("--roi-point", action="append", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--roi-max-track-gap-frames", type=int, default=30)
    parser.add_argument("--gt-person-classes", nargs="+", type=int, default=[1])
    parser.add_argument("--gt-min-visibility", type=float, default=0.0)
    parser.add_argument("--include-unmarked-gt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run shared-algorithm MOT evaluation and return a process exit code."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logger = configure_logging(level="INFO")
    try:
        output_dir = cast(Path, args.output_dir).expanduser().resolve()
        _prepare_outputs(output_dir, overwrite=cast(bool, args.overwrite))
        line = cast(list[float], args.counting_line)
        counting = CountingSettings(
            enabled=True,
            line=CountingLineSettings(
                p1=(line[0], line[1]),
                p2=(line[2], line[3]),
                enter_side=cast(SideName, args.enter_side),
            ),
            min_track_age=cast(int, args.min_track_age),
            min_displacement_pixels=cast(float, args.min_displacement_pixels),
            cooldown_frames=cast(int, args.cooldown_frames),
            max_track_gap_frames=cast(int, args.max_track_gap_frames),
        )
        roi_points = cast(list[list[float]] | None, args.roi_point)
        roi = RoiSettings()
        if roi_points is not None:
            roi = RoiSettings(
                enabled=True,
                name=cast(str, args.roi_name),
                points=tuple((point[0], point[1]) for point in roi_points),
                max_track_gap_frames=cast(int, args.roi_max_track_gap_frames),
            )
        policy = GroundTruthPolicy(
            person_class_ids=tuple(cast(list[int], args.gt_person_classes)),
            require_marked=not cast(bool, args.include_unmarked_gt),
            min_visibility=cast(float, args.gt_min_visibility),
        )
        ground_truth = load_mot_ground_truth(cast(Path, args.mot_sequence))
        predicted_records = read_track_records_csv(cast(Path, args.predicted_tracks))
        evaluation = evaluate_mot_prediction(
            ground_truth,
            predicted_records,
            counting=counting,
            roi=roi,
            ground_truth_policy=policy,
        )
        artifacts = _write_evaluation_outputs(output_dir, evaluation)
        payload = {
            **evaluation.as_dict(),
            "mot_sequence": str(cast(Path, args.mot_sequence).expanduser().resolve()),
            "predicted_tracks": str(cast(Path, args.predicted_tracks).expanduser().resolve()),
            "counting": counting.model_dump(mode="json"),
            "roi": roi.model_dump(mode="json"),
            "artifacts": artifacts,
        }
        write_json(output_dir / "evaluation.json", payload)
    except (ValidationError, PeopleFlowError) as exc:
        logger.error("Evaluation failed: %s", exc)
        return 2
    logger.info(
        "Evaluation complete: sequence=%s count_mae=%.3f occupancy_mae=%s output=%s",
        evaluation.sequence_name,
        evaluation.metrics.count_mae,
        evaluation.metrics.occupancy_mae,
        output_dir,
    )
    return 0


def _prepare_outputs(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in _OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise EvaluationError(f"Evaluation outputs already exist: {names}. Use --overwrite.")


def _write_evaluation_outputs(
    output_dir: Path,
    evaluation: MotPredictionEvaluation,
) -> dict[str, str | None]:
    ground_truth_events = output_dir / "ground_truth_events.csv"
    predicted_events = output_dir / "predicted_events.csv"
    with EventCsvWriter(ground_truth_events) as writer:
        writer.write(list(evaluation.ground_truth.events))
    with EventCsvWriter(predicted_events) as writer:
        writer.write(list(evaluation.predicted.events))

    artifacts: dict[str, str | None] = {
        "ground_truth_events": ground_truth_events.name,
        "predicted_events": predicted_events.name,
        "ground_truth_occupancy": None,
        "predicted_occupancy": None,
        "ground_truth_dwell_times": None,
        "predicted_dwell_times": None,
    }
    if evaluation.ground_truth.occupancy:
        artifacts.update(_write_roi_outputs(output_dir, "ground_truth", evaluation.ground_truth))
        artifacts.update(_write_roi_outputs(output_dir, "predicted", evaluation.predicted))
    return artifacts


def _write_roi_outputs(
    output_dir: Path,
    prefix: str,
    analysis: TrajectoryAnalysis,
) -> dict[str, str]:
    occupancy_path = output_dir / f"{prefix}_occupancy.csv"
    with OccupancyCsvWriter(occupancy_path) as writer:
        for snapshot in analysis.occupancy:
            writer.write(snapshot)
    dwell_path = output_dir / f"{prefix}_dwell_times.csv"
    write_dwell_times_csv(dwell_path, analysis.dwell_records)
    return {
        f"{prefix}_occupancy": occupancy_path.name,
        f"{prefix}_dwell_times": dwell_path.name,
    }


if __name__ == "__main__":
    raise SystemExit(main())
