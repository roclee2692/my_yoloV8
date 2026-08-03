# Phase 8 baseline experiment results

Date: 2026-08-02 (Asia/Shanghai)
Branch: `v2/yolo26-people-flow`

## Evidence boundary

These are real local RTX 4060 measurements on the verified Ultralytics sample. The
sample has no associated Ground Truth. Therefore ID switches, counting errors,
count MAE/MAPE, and occupancy MAE are unavailable (`null`), not zero. Predicted
counts may be compared for behavioral differences but cannot be ranked by accuracy.

## Controlled environment and input

- Interpreter: user-managed `ai_env`, Python 3.10.19
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- PyTorch: 2.6.0+cu124; CUDA available
- Ultralytics: 8.4.37
- OpenCV: 4.13.0
- Input: verified `People-counting-compressed.mp4`
- Input metadata: 250 frames, 50 FPS, 1280x720
- Class: COCO person class 0 only
- Device / seed: CUDA device 0 / 42
- Confidence / IoU / image size: 0.25 / 0.70 / 640
- Directed line: `(100,500) -> (1100,500)`, enter side positive
- ROI: `(100,100), (1100,100), (1100,650), (100,650)`

Both model weights and the sample matched the repository's pinned official size and
SHA-256 metadata before execution. The user-managed environment has Ultralytics
8.4.37 while the project dependency is pinned to 8.4.114; this version difference is
retained as a limitation rather than hidden.

## Command

```powershell
cd F:\my_yoloV8
$env:PYTHONPATH = (Resolve-Path src).Path
& "C:\Users\Raelon\miniconda3\envs\ai_env\python.exe" `
  scripts\run_baseline.py `
  --config configs\experiments\baseline.yaml `
  --project-root .
```

## Measured results

| Metric | A: v8n + ByteTrack | B: 26n + ByteTrack | C: 26n + BoT-SORT |
|---|---:|---:|---:|
| Processed frames | 250 | 250 | 250 |
| Track observations | 349 | 222 | 226 |
| Unique Track IDs | 6 | 6 | 5 |
| Line enter / exit | 1 / 1 | 0 / 1 | 0 / 1 |
| ROI enter / observed exit | 6 / 1 | 6 / 1 | 5 / 1 |
| Maximum occupancy | 2 | 2 | 2 |
| Average occupancy | 0.924 | 0.728 | 0.744 |
| End-to-end processing FPS | 18.364 | 26.159 | 17.980 |
| Average frame latency (ms) | 32.993 | 34.649 | 52.723 |
| P95 frame latency (ms) | 35.958 | 44.795 | 65.436 |
| Peak GPU memory (MiB) | 33.362 | 75.464 | 85.256 |
| Total runtime (s) | 13.614 | 9.557 | 13.904 |
| Ground Truth available | no | no | no |
| ID switches | unavailable | unavailable | unavailable |
| Count / occupancy errors | unavailable | unavailable | unavailable |

`Track observations` is the number of per-frame persisted person rows. It is not a
count of unique real people. `Unique Track IDs` is tracker output and can be affected
by fragmentation or identity reuse.

## Interpretation

### YOLO26n versus YOLOv8n with ByteTrack

The two runs produced different trajectories and line totals: A produced two line
events while B produced one. B produced 36.4% fewer Track observations, the same six
unique IDs, and lower average ROI occupancy. Without Ground Truth, none of these
differences proves an improvement or regression in final counting accuracy.

B's end-to-end FPS was 42.4% higher and total runtime 29.8% lower than A, but its
average per-frame latency was 5.0% higher. This apparent tension is expected because
end-to-end FPS includes initialization and the fixed experiment order gave A the
first CUDA/model cold start. The single run does not establish that YOLO26n is
intrinsically faster.

### BoT-SORT versus ByteTrack with YOLO26n

B and C produced the same line prediction (0 enter, 1 exit). C produced one fewer
unique Track ID and one fewer ROI entry, but correctness is unknown. Relative to B,
C had 31.3% lower end-to-end FPS, 52.2% higher average latency, 46.1% higher P95
latency, 13.0% higher peak GPU memory, and 45.5% longer total runtime.

On this sample, BoT-SORT has measurable extra cost and no GT-verified accuracy gain.
That is insufficient to declare it worthwhile; the decision must wait for MOT17 or
another labeled sequence.

## Independent artifact verification

- Each annotated MP4 independently decoded to 250 frames at 50 FPS.
- All three `tracks.csv` files contained only class 0.
- Track rows / unique IDs were 349/6, 222/6, and 226/5.
- Every event evidence JPEG existed and decoded successfully.
- Every `occupancy.csv` contained exactly 250 rows.
- Recomputed maximum/average occupancy matched each `summary.json` and comparison row.
- Every `evaluation.json` stated `ground_truth_available: false` and stored all GT
  error fields and `id_switches` as JSON `null`.
- Effective configs matched the shared source, thresholds, geometry, device, seed,
  and captured software versions.

## Automated quality gates

The reproducible project `.venv` produced `71 passed` from `pytest -q` after the
Phase 8 changes. `ruff check .`, strict `mypy src tests scripts`, compileall, and
`pip check` also completed successfully. CI remains model-free and does not download
weights or require a GPU.

## Failure cases and limitations

- The official sample is one short scene and contains pre-existing visual overlays.
- No MOT17 data was present locally, so accuracy and identity stability are unknown.
- One sequential trial does not control cold-start, thermal, or scheduling variance.
- Track disappearance and ID fragmentation can change unique-ID, ROI, and dwell values.
- Multiple crowd densities, camera angles, occlusion levels, and lighting conditions
  were not evaluated.

## Answers to the Phase 8 questions

1. **Does YOLO26 reduce final counting error versus YOLOv8?** Unknown. This source
   has no Ground Truth, so the required error comparison is unavailable.
2. **Is BoT-SORT worth its extra compute versus ByteTrack?** Not demonstrated here.
   It cost materially more runtime, latency, and GPU memory, while no accuracy gain
   can be verified without Ground Truth.

No training decision should be made from this sample alone. The next evidence step
is to supply official MOT17, convert a selected sequence, run the same matrix, and
populate the currently null error and ID-switch columns.
