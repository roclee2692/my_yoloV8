# Phase 6 local GPU smoke-test evidence

Date: 2026-08-02 (Asia/Shanghai)
Branch: `v2/yolo26-people-flow`

## Evidence boundary

This is a real local GPU functional smoke test of YOLO26n, ByteTrack, Phase 5 directional counting, and Phase 6 polygon ROI analytics. The sample has no Ground Truth. The observed transitions, occupancy, and dwell values are model predictions, not accuracy measurements.

## Environment

- Explicit interpreter: `C:\Users\Raelon\miniconda3\envs\ai_env\python.exe`
- Python: 3.10.19
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- NVIDIA driver: 572.83
- PyTorch: 2.6.0+cu124
- `torch.cuda.is_available()`: `True`
- Ultralytics: 8.4.37
- OpenCV: 4.13.0

The project pins Ultralytics 8.4.114; this user-managed runtime has 8.4.37. Phase 5 already documented the Ultralytics AutoUpdate that installed LAP 0.5.13 and changed NumPy to 2.2.6. No package installation or update occurred during this Phase 6 run.

## Input and configuration

- Source: `data/samples/People-counting-compressed.mp4`
- Source metadata: 250 frames, 50 FPS, 1280x720
- Model / tracker: YOLO26n / ByteTrack
- Class filter: COCO person class 0
- Device: CUDA device 0, without CPU fallback
- Counting line: `(100, 500) -> (1100, 500)`, enter side `positive`
- ROI: `entrance_area`
- ROI points: `(100,100)`, `(1100,100)`, `(1100,650)`, `(100,650)`
- ROI maximum retained Track gap: 30 frames

## Command

```powershell
cd F:\my_yoloV8
$env:PYTHONPATH = (Resolve-Path src).Path

& "C:\Users\Raelon\miniconda3\envs\ai_env\python.exe" `
  -m people_flow.cli run `
  --source data\samples\People-counting-compressed.mp4 `
  --model yolo26n.pt `
  --tracker bytetrack.yaml `
  --classes 0 `
  --device 0 `
  --output-dir runs\phase6_gpu `
  --counting-line 100 500 1100 500 `
  --enter-side positive `
  --min-track-age 5 `
  --min-displacement-pixels 15 `
  --cooldown-frames 30 `
  --max-track-gap-frames 30 `
  --roi-name entrance_area `
  --roi-point 100 100 `
  --roi-point 1100 100 `
  --roi-point 1100 650 `
  --roi-point 100 650 `
  --roi-max-track-gap-frames 30
```

## Measured outputs

| Measurement | Observed value |
|---|---:|
| Input / processed / decoded output frames | 250 / 250 / 250 |
| Source / output FPS | 50.0 / 50.0 |
| Track rows / unique IDs | 222 / 6 |
| Recorded classes | 0 only |
| Line enter / exit | 0 / 1 |
| ROI total enter / observed exit | 6 / 1 |
| Maximum / average occupancy | 2 / 0.728 |
| First peak frame / time | 11 / 200.0 ms |
| Tracks with dwell | 5 |
| Inside at end | 2 |
| Average / median dwell | 0.992 / 0.66 s |
| Processing FPS | 11.649163720515052 |
| Average / P95 latency | 45.98490600014338 / 46.85409999729018 ms |
| Peak GPU memory | 63.55126953125 MiB |
| Total runtime | 21.460768000000826 s |

Observed ROI transitions occurred at frames 1, 11, 19, 78, 81, 87, and 223. Track 16 was observed entering at frame 78, exiting at frame 81, and re-entering at frame 87. Other Tracks that disappeared while last observed inside were not falsely labeled as exits.

## Independent artifact verification

- `occupancy.csv` contained exactly 250 rows.
- Recomputing its occupancy column produced maximum `2` and average `0.728`.
- Every Track-ID cell parsed as valid JSON.
- `dwell_times.csv` contained five rows; recomputed mean and median were `0.992` and `0.66` seconds.
- `tracks.csv` contained 222 rows, six unique IDs, and only class 0.
- `events.csv` contained the same single Phase 5 exit event.
- OpenCV decoded `annotated.mp4` to 250 frames at 50 FPS.
- The event evidence JPEG decoded as a 1280x720 three-channel image.

Generated files under ignored `runs/phase6_gpu/`:

```text
annotated.mp4                                      5,239,243 bytes
tracks.csv                                           39,468 bytes
events.csv                                              280 bytes
occupancy.csv                                         9,538 bytes
dwell_times.csv                                         461 bytes
summary.json                                            370 bytes
run_config.yaml                                         869 bytes
runtime_metrics.json                                    369 bytes
run.log                                                 907 bytes
evidence/event_000001_frame_000179_track_16_exit.jpg 171,752 bytes
```

## Automated verification

The reproducible project `.venv` produced `59 passed` from `pytest -q`; `ruff check .`, strict mypy, compileall, and `pip check` also succeeded in the final local verification. CI performs the model-free quality gates without downloading weights or requiring a GPU.

## Limitations

- No Ground Truth is available, so ROI entries, exits, occupancy, and dwell cannot be labeled correct or incorrect.
- A Track disappearing while inside is closed at its last observation after the configured gap, but is not counted as an observed exit.
- Tracker ID changes can split one person into multiple dwell records.
- The source contains pre-existing demonstration overlays.
- One short sample and one ROI do not constitute a controlled benchmark.
