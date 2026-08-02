# Phase 7 synthetic Ground Truth validation

Date: 2026-08-02 (Asia/Shanghai)

## Purpose and evidence boundary

This deterministic two-frame MOT fixture validates the evaluator, schemas, metric
formulas, and shared-algorithm requirement without a detector, tracker, GPU, or
downloaded dataset. Its values are test expectations, not YOLOv8, YOLO26,
ByteTrack, or BoT-SORT performance results.

## Fixture

- Frame rate: 10 FPS
- Frame count: 2
- Directed line: `(0, 5) -> (10, 5)`, enter side `positive`
- ROI: rectangle `(0, 5), (10, 5), (10, 20), (0, 20)`
- Thresholds: track age 2, displacement 5 pixels, cooldown 1 frame
- Ground Truth: one marked pedestrian crosses the line and enters the ROI
- Prediction: one person Track remains on the original side and outside the ROI
- Distractor and unmarked MOT rows: rejected by the default Ground Truth policy

## Expected and observed assertions

| Metric | Value |
|---|---:|
| Ground Truth enter / exit | 1 / 0 |
| Predicted enter / exit | 0 / 0 |
| Enter / exit absolute error | 1 / 0 |
| Total count absolute error | 1 |
| Count MAE | 0.5 |
| Count MAPE | 100.0% |
| Occupancy MAE | 0.5 |

The integration test also confirms that the CLI creates Ground Truth and predicted
event, occupancy, and dwell CSV files plus `evaluation.json`; refuses to overwrite
them by default; and succeeds with explicit `--overwrite`.

## Command

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests\integration\test_ground_truth_evaluation.py `
  tests\unit\test_counting_metrics.py `
  tests\unit\test_trajectory_evaluator.py
```

Observed result: `6 passed`.

## Limitation

`data/raw/MOT17` was absent during this validation. Real MOT17 metrics remain
unavailable until the user supplies the official dataset and runs prediction on the
corresponding sequence with the same line and ROI configuration.
