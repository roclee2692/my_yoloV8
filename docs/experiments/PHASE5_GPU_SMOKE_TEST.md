# Phase 5 local GPU smoke-test evidence

Date: 2026-08-02 (Asia/Shanghai)
Branch: `v2/yolo26-people-flow`

## Evidence boundary

This is a real local GPU functional smoke test of YOLO26n, ByteTrack, directional counting, and artifact output. The sample video has no Ground Truth. The observed event count is therefore **not** a counting-accuracy measurement and must not be used as a Phase 8 model or tracker comparison.

## Environment

- Explicit interpreter: `C:\Users\Raelon\miniconda3\envs\ai_env\python.exe`
- Python: 3.10.19
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- NVIDIA driver: 572.83
- PyTorch: 2.6.0+cu124
- `torch.cuda.is_available()`: `True`
- Ultralytics: 8.4.37
- OpenCV: 4.13.0
- NumPy after Ultralytics AutoUpdate: 2.2.6
- LAP after Ultralytics AutoUpdate: 0.5.13

The project pins Ultralytics 8.4.114 in `pyproject.toml`; this user-managed environment contains 8.4.37. The run verifies the stated environment only and does not prove equivalence with the pinned reproducible environment.

## Input and configuration

- Source: `data/samples/People-counting-compressed.mp4`
- Source metadata: 250 frames, 50 FPS, 1280x720
- Model: `weights/yolo26n.pt`
- Tracker: `bytetrack.yaml`
- Class filter: COCO person class 0
- Device: CUDA device 0; no CPU fallback
- Directed line: `(100, 500) -> (1100, 500)`
- Enter side: `positive`
- Minimum track age: 5 observations
- Minimum displacement: 15 pixels
- Cooldown: 30 frames
- Maximum retained track gap: 30 frames

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
  --output-dir runs\phase5_gpu `
  --counting-line 100 500 1100 500 `
  --enter-side positive `
  --min-track-age 5 `
  --min-displacement-pixels 15 `
  --cooldown-frames 30 `
  --max-track-gap-frames 30 `
  --overwrite
```

## Measured outputs

| Measurement | Observed value |
|---|---:|
| Input / processed frames | 250 / 250 |
| Decoded output frames | 250 |
| Source / output FPS | 50.0 / 50.0 |
| Track rows | 222 |
| Unique Track IDs | 6 |
| Recorded classes | 0 only |
| Enter / exit events | 0 / 1 |
| Processing FPS | 11.647099120847816 |
| Average per-frame latency | 43.72682879999775 ms |
| Nearest-rank P95 latency | 42.47679999934917 ms |
| Peak GPU memory reported by PyTorch | 63.55126953125 MiB |
| Total runtime | 21.464572200000475 s |

The single event was:

```text
event_000001, track 16, exit, frame 179, timestamp 3560.0 ms,
positive -> negative, foot point (635.7476, 497.1413), confidence 0.2730
```

The event points to `evidence/event_000001_frame_000179_track_16_exit.jpg`. An independent OpenCV read decoded that file as a 1280x720 three-channel image. The annotated MP4 was independently decoded to all 250 frames at 50 FPS.

Generated files under ignored `runs/phase5_gpu/`:

```text
annotated.mp4                                      4,836,395 bytes
tracks.csv                                           39,468 bytes
events.csv                                              280 bytes
run_config.yaml                                         678 bytes
runtime_metrics.json                                    369 bytes
run.log                                                 796 bytes
evidence/event_000001_frame_000179_track_16_exit.jpg 157,058 bytes
```

## First-attempt failure retained as evidence

The first GPU process was not counted as successful. Ultralytics found that `lap>=0.5.12` was absent and triggered its own AutoUpdate, installing LAP 0.5.13 and changing NumPy 2.2.5 to 2.2.6. That process then exited with Intel OpenMP error 15 and explicitly requested a process restart. It left only partial zero-length CSV/video outputs.

The runtime was restarted without setting the unsafe `KMP_DUPLICATE_LIB_OK=TRUE` workaround. The second command used explicit `--overwrite`, completed normally, and replaced the partial artifacts. No other package was installed for this phase.

## Automated verification

The reproducible project `.venv` produced `50 passed` from `pytest -q`; `ruff check .`, strict mypy, compileall, and `pip check` also succeeded. The user-managed `ai_env` does not contain pytest, so tests were not claimed in that environment and no test package was installed into it.

`python -m pip check` in `ai_env` reported a pre-existing, project-unrelated conflict: `pdfplumber 0.11.9` requires `pdfminer.six==20251230`, while that environment has `pdfminer-six 20250506`. It did not prevent the GPU run but remains an environment maintenance issue.

## Limitations

- No Ground Truth is available for the sample, so the event cannot be labeled correct or incorrect.
- The source already contains demonstration overlays; the output combines those pixels with this project's annotation.
- One short sample and one configuration do not establish model accuracy, tracker quality, or stable performance.
- Phase 7 must generate Ground Truth events through the same counter before count error can be measured.
