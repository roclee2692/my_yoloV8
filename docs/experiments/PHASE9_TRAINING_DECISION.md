# Phase 9 training-readiness decision

Date: 2026-08-03 (Asia/Shanghai)
Branch: `v2/yolo26-people-flow`

## Decision

**Training is deferred. No model training or fine-tuning was started.**

The Phase 8 A/B/C baseline contains three real GPU runs, but the sample has no
associated Ground Truth. ID switches, count MAE/MAPE, and occupancy MAE are null.
Detection Recall was not measured, and no evidence attributes the observed
prediction differences primarily to missed detections or occlusion.

This means none of the four permitted Phase 9 triggers has been demonstrated:

1. obvious YOLO26 pretrained-model missed detections;
2. detection Recall proven to be the main source of counting error;
3. detection shown to be inadequate in occlusion scenes;
4. baseline data proving that fine-tuning is necessary.

Starting a 30-epoch run here would violate the project requirement against training
for demonstration purposes.

## Reproducible gate

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& ".venv\Scripts\python.exe" `
  scripts\assess_training_readiness.py `
  --comparison runs\experiments\comparison.csv `
  --output runs\training\phase9_readiness.json
```

The command treats a valid `deferred` decision as successful execution and writes a
structured JSON record. Existing output is protected unless `--overwrite` is used.

## Current evidence

| Evidence | Result |
|---|---:|
| Baseline experiments | 3 |
| Runs with Ground Truth | 0 |
| Runs with measured count and occupancy error | 0 |
| Detection Recall available | no |
| Detection-to-count-error attribution available | no |
| Occlusion-specific failure evidence available | no |
| Training allowed | no |

## Required next evidence

Before training can be reconsidered:

1. supply an official MOT17 training sequence;
2. run the unchanged Phase 8 matrix on the corresponding video;
3. measure detection Recall and counting/occupancy errors;
4. examine whether missed detections, tracking identity errors, or counting geometry
   is the dominant source;
5. approve YOLO26n fine-tuning only if the evidence selects detection quality as the
   actionable bottleneck.

No weights, fabricated curves, synthetic training logs, or unverified validation
metrics were created in Phase 9.
