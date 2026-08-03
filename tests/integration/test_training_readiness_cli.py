"""Integration test for the Phase 9 training-readiness command."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from people_flow.evaluation.training_readiness_cli import main


def test_readiness_cli_writes_deferred_decision_and_protects_output(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    output = tmp_path / "phase9_readiness.json"
    with comparison.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "experiment_id",
                "ground_truth_available",
                "id_switches",
                "count_mae",
                "occupancy_mae",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "A",
                "ground_truth_available": "False",
                "id_switches": "",
                "count_mae": "",
                "occupancy_mae": "",
            }
        )

    argv = ["--comparison", str(comparison), "--output", str(output)]
    assert main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"] == "deferred"
    assert payload["training_allowed"] is False
    assert main(argv) == 2
    assert main([*argv, "--overwrite"]) == 0
