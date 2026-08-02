# Phase 8 controlled experiment matrix

## Scope

Phase 8 runs a strict YAML-defined suite and changes only the model/tracker pair
between experiments. Detection, tracking, counting, ROI, output, and Ground Truth
logic remain in the production modules; the suite runner only validates shared
settings, invokes `run_pipeline`, evaluates persisted Tracks, and aggregates results.

## Required matrix

| ID | Model | Tracker |
|---|---|---|
| A | YOLOv8n | ByteTrack |
| B | YOLO26n | ByteTrack |
| C | YOLO26n | BoT-SORT |

The repository configuration fixes one source, device, class filter, confidence,
IoU, image size, seed, virtual line, ROI, and all counter thresholds for the entire
suite. Per-experiment overrides for these fields are intentionally unsupported.

## Command

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& "C:\Users\<USER>\miniconda3\envs\ai_env\python.exe" `
  scripts\run_baseline.py `
  --config configs\experiments\baseline.yaml `
  --project-root .
```

Use the Python interpreter appropriate for the local machine. Existing experiment
outputs are protected unless `--overwrite` is explicit.

## Per-experiment outputs

```text
runs/experiments/<experiment_id>/
├── config.yaml
├── annotated.mp4
├── tracks.csv
├── events.csv
├── occupancy.csv
├── summary.json
├── runtime_metrics.json
├── evaluation.json
└── run.log
```

The pipeline also retains `run_config.yaml`, `dwell_times.csv`, and event evidence
JPEGs. `config.yaml` merges the effective pipeline configuration and captured
software versions with the suite ID, experiment ID, and seed.

## Comparison semantics

`runs/experiments/comparison.csv` distinguishes:

- `track_observations`: person Track rows across all frames, not unique people;
- `unique_track_ids`: distinct tracker identities, not guaranteed real people;
- directional line predictions and ROI summaries;
- end-to-end FPS/runtime, per-frame latency, and peak GPU memory;
- GT availability, ID switches, count errors, and occupancy error.

When no corresponding MOT Ground Truth is configured, ID switches and every error
metric are empty in CSV and JSON `null`, with an explicit reason. They are never
silently treated as zero.

## Tracking metric boundary

When Ground Truth is present, identity matching uses a documented deterministic
`continuity_first_greedy_iou` method. A valid prior GT-to-predicted association is
retained before remaining boxes are matched one-to-one in descending IoU order. An
association changing to another predicted ID increments `id_switches`.

This transparent project metric is useful for controlled comparisons, but it is not
labeled as the official MOTChallenge evaluation result. The matching method and IoU
threshold are serialized beside every measured value.

## Reproducibility limits

One sequential run is an initial baseline, not a statistical performance study.
End-to-end `processing_fps` includes model initialization and output finalization,
while average/P95 latency measures the per-frame processing loop. Cold CUDA context,
OS scheduling, thermals, and experiment order can affect the former. Strong speed
claims require repeated cold and warm trials with randomized order.
