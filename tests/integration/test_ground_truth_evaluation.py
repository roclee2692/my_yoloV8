"""Model-free end-to-end MOT Ground Truth evaluation test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evaluate_experiment import main

from people_flow.outputs.csv_writer import TrackCsvWriter
from people_flow.tracking.track_record import TrackRecord


def _create_sequence(root: Path) -> Path:
    sequence = root / "MOT17-99-SDP"
    (sequence / "img1").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "seqinfo.ini").write_text(
        """[Sequence]
name=MOT17-99-SDP
imDir=img1
frameRate=10
seqLength=2
imWidth=100
imHeight=50
imExt=.jpg
""",
        encoding="utf-8",
    )
    (sequence / "gt" / "gt.txt").write_text(
        """1,1,4,1,2,4,1,1,1
2,1,4,11,2,4,1,1,1
""",
        encoding="utf-8",
    )
    return sequence


def _write_predictions(path: Path) -> None:
    records = [
        TrackRecord.from_bbox(
            frame_id=frame_id,
            timestamp_ms=(frame_id - 1) * 100.0,
            track_id=10,
            class_id=0,
            confidence=0.9,
            bbox=(4.0, 1.0, 6.0, 5.0),
        )
        for frame_id in (1, 2)
    ]
    with TrackCsvWriter(path) as writer:
        writer.write(records)


def test_evaluate_experiment_cli_writes_auditable_outputs(tmp_path: Path) -> None:
    """CLI should write GT/prediction evidence and protect existing outputs."""

    sequence = _create_sequence(tmp_path)
    tracks = tmp_path / "tracks.csv"
    output = tmp_path / "evaluation"
    _write_predictions(tracks)
    argv = [
        "--mot-sequence",
        str(sequence),
        "--predicted-tracks",
        str(tracks),
        "--output-dir",
        str(output),
        "--counting-line",
        "0",
        "10",
        "20",
        "10",
        "--min-track-age",
        "2",
        "--min-displacement-pixels",
        "5",
        "--cooldown-frames",
        "0",
        "--max-track-gap-frames",
        "2",
        "--roi-name",
        "inside",
        "--roi-point",
        "0",
        "10",
        "--roi-point",
        "20",
        "10",
        "--roi-point",
        "20",
        "20",
        "--roi-point",
        "0",
        "20",
    ]

    assert main(argv) == 0
    expected = {
        "ground_truth_events.csv",
        "predicted_events.csv",
        "ground_truth_occupancy.csv",
        "predicted_occupancy.csv",
        "ground_truth_dwell_times.csv",
        "predicted_dwell_times.csv",
        "evaluation.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    with (output / "ground_truth_events.csv").open(encoding="utf-8", newline="") as stream:
        truth_events = list(csv.DictReader(stream))
    with (output / "predicted_events.csv").open(encoding="utf-8", newline="") as stream:
        predicted_events = list(csv.DictReader(stream))
    assert [event["event_type"] for event in truth_events] == ["enter"]
    assert predicted_events == []

    payload = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["ground_truth_available"] is True
    assert payload["ground_truth_total"] == 1
    assert payload["predicted_total"] == 0
    assert payload["count_mae"] == 0.5
    assert payload["count_mape"] == 100.0
    assert payload["occupancy_mae"] == 0.5
    assert main(argv) == 2
    assert main([*argv, "--overwrite"]) == 0
