# Phase 3 local smoke-test evidence

Date: 2026-08-01 (Asia/Shanghai)

These are real local smoke-test outputs for pipeline acceptance, not a controlled benchmark and not a Ground Truth evaluation. They must not be used to conclude that one model or tracker is more accurate than another.

## Environment

- OS: Windows
- CPU: AMD Ryzen 7 8845H, 8 cores / 16 logical processors
- CUDA: unavailable (`torch.cuda.is_available() == False`)
- Python: 3.13.12
- Ultralytics: 8.4.114
- PyTorch: 2.13.0+cpu
- OpenCV: 4.14.0
- NumPy: 2.2.6

## Verified local assets

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `yolo26n.pt` | 5,544,453 | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| `yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `People-counting-compressed.mp4` | 358,219 | `e9ebad02081416465caae6ff87a28f96c441337742266b45f29a8c8a7b80c3a5` |

The source video contains 250 frames at 50 FPS and 1280x720. All assets came from the official `ultralytics/assets` GitHub releases and remain ignored by Git.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\download_phase3_assets.py

.\.venv\Scripts\people-flow.exe run `
  --source data\samples\People-counting-compressed.mp4 `
  --model yolo26n.pt `
  --tracker bytetrack.yaml `
  --classes 0 `
  --device cpu `
  --output-dir runs\demo `
  --overwrite

.\.venv\Scripts\python.exe -m people_flow.cli run `
  --source data\samples\People-counting-compressed.mp4 `
  --model yolo26n.pt `
  --tracker botsort.yaml `
  --classes 0 `
  --device cpu `
  --output-dir runs\module_botsort
```

The YOLOv8n switch was also exercised with the console command, `yolov8n.pt`, ByteTrack, and `runs/smoke_yolov8n`.

## Observed outputs

| Model | Tracker | Frames in/out | Track rows | Unique IDs | Output FPS | Processing FPS | Avg latency ms | P95 latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n | ByteTrack | 250 / 250 | 221 | 6 | 50.0 | 12.327 | 65.138 | 69.953 |
| YOLO26n | BoT-SORT | 250 / 250 | 225 | 5 | 50.0 | 9.316 | 91.217 | 97.166 |
| YOLOv8n | ByteTrack | 250 / 250 | 349 | 6 | 50.0 | 5.836 | 155.999 | 62.982 |

Each run produced `annotated.mp4`, `tracks.csv`, `run_config.yaml`, `runtime_metrics.json`, and `run.log`. Every recorded class was `0` (person). The YOLOv8n latency includes a large first-frame initialization outlier, which is why its average can exceed its nearest-rank P95 on this short run.

## Limitations

- No NVIDIA/CUDA device was available, so CUDA performance was not measured.
- There is no Ground Truth for this sample; no tracking or counting accuracy is claimed.
- The official sample already contains demonstration overlays, so the generated video has overlapping source and People Flow annotations; a decoded frame was inspected to confirm our green boxes, IDs, confidence labels, and foot points were present.
- The runs were sequential smoke tests, not a controlled Phase 8 experiment matrix.
- Runtime outputs are intentionally not committed; the values above were transcribed from their generated JSON/CSV files after validation.
