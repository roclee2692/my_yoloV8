# Data policy

The repository does not distribute raw datasets or input videos.

- `raw/`: user-downloaded source datasets such as MOT17; never committed.
- `interim/`: converted videos and intermediate artifacts; never committed.
- `samples/`: local short input media for integration runs; never committed by default.

Only `.gitkeep` placeholders are versioned. Phase 4 will document the official MOT17 acquisition method, expected directory structure, license considerations, and integrity checks before MOT data support is implemented.

数据默认不随仓库分发。请将自行获取的原始数据放入 `raw/`，转换产物放入 `interim/`，本地短视频放入 `samples/`。这些内容均由 `.gitignore` 排除。

## Phase 3 official sample

The optional local smoke-test video is an Ultralytics official release asset and is not committed:

- name: `People-counting-compressed.mp4`
- source: `https://github.com/ultralytics/assets/releases/download/v0.0.0/People-counting-compressed.mp4`
- size: `358219` bytes
- SHA-256: `e9ebad02081416465caae6ff87a28f96c441337742266b45f29a8c8a7b80c3a5`

Acquire and verify it with `python scripts/download_phase3_assets.py`. CI creates its own synthetic MP4 and never downloads this sample or a model.

This official clip already contains a counting line, boxes, and labels from its original demonstration. It is suitable for repeatable pipeline smoke tests but produces visually overlapping annotations. Phase 4 will create clean evaluation videos from original MOT17 image sequences.
