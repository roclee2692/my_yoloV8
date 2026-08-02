# Phase 7 MOT Ground Truth evaluation

## Scope

Phase 7 converts MOT Ground Truth and predicted `tracks.csv` rows into the same
canonical `TrackRecord` representation, feeds both streams through the production
`LineCounter` and `RoiCounter`, and computes counting and occupancy errors. It does
not introduce a second Ground Truth-only counting implementation.

## Shared evaluation path

```text
MOT gt.txt                         predicted tracks.csv
    |                                      |
    +--> validated person TrackRecord <----+
                         |
             identical frame timestamps
                         |
             +-----------+-----------+
             |                       |
        LineCounter              RoiCounter
             |                       |
       directional events       frame occupancy
             +-----------+-----------+
                         |
                  counting metrics
```

MOT frame timestamps are derived as `(frame_id - 1) * 1000 / frameRate` from
`seqinfo.ini`. Input timestamps therefore cannot give Ground Truth and predictions
different timing semantics.

## Ground Truth policy

The default MOT17 policy keeps only:

- class ID `1` (pedestrian), then normalizes it to COCO person class `0`;
- marked rows (`confidence`/mark greater than zero);
- rows whose visibility is at least `0.0`.

The class list, minimum visibility, and marked-row policy are explicit CLI options
and are serialized into `evaluation.json`. Bounding boxes use the same bottom-center
foot-point calculation as prediction records.

## Metrics

- `enter_absolute_error = abs(predicted_enter - ground_truth_enter)`
- `exit_absolute_error = abs(predicted_exit - ground_truth_exit)`
- `total_count_absolute_error = abs(predicted_total - ground_truth_total)`
- `count_mae = (enter_absolute_error + exit_absolute_error) / 2`
- `count_mape = total_count_absolute_error / ground_truth_total * 100`
- `occupancy_mae` is the mean per-frame absolute occupancy difference.

When Ground Truth total count is zero, `count_mape` is JSON `null`; the evaluator
never divides by zero or substitutes a misleading percentage. Occupancy arrays must
have the same frame count or evaluation stops with an explicit error.

## Command

```bash
python scripts/evaluate_experiment.py \
  --mot-sequence data/raw/MOT17/train/MOT17-04-SDP \
  --predicted-tracks runs/MOT17-04-SDP/tracks.csv \
  --output-dir runs/MOT17-04-SDP/evaluation \
  --counting-line 100 500 1100 500 \
  --enter-side positive \
  --min-track-age 5 \
  --min-displacement-pixels 15 \
  --cooldown-frames 30 \
  --max-track-gap-frames 30 \
  --roi-name entrance_area \
  --roi-point 100 100 \
  --roi-point 1100 100 \
  --roi-point 1100 650 \
  --roi-point 100 650
```

Existing evaluation artifacts are protected unless `--overwrite` is supplied.

## Outputs

```text
ground_truth_events.csv
predicted_events.csv
ground_truth_occupancy.csv
predicted_occupancy.csv
ground_truth_dwell_times.csv
predicted_dwell_times.csv
evaluation.json
```

ROI files are emitted only when an ROI is configured. Event evidence paths are
preserved in the schema, but MOT text annotations have no source evidence JPEG and
therefore leave that value empty.

## Verification boundary

Unit tests cover filtering, shared counter behavior, strict Track CSV parsing,
zero-safe MAPE, occupancy MAE, and unequal frame arrays. A model-free integration
test invokes the actual CLI on a synthetic MOT-format sequence and verifies all
seven artifacts plus overwrite protection. Official MOT17 data is not bundled and
was not available locally during Phase 7, so no real-model accuracy claim is made.
