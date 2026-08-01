# Data policy

The repository does not distribute raw datasets or input videos.

- `raw/`: user-downloaded source datasets such as MOT17; never committed.
- `interim/`: converted videos and intermediate artifacts; never committed.
- `samples/`: local short input media for integration runs; never committed by default.

Only `.gitkeep` placeholders are versioned. Phase 4 will document the official MOT17 acquisition method, expected directory structure, license considerations, and integrity checks before MOT data support is implemented.

数据默认不随仓库分发。请将自行获取的原始数据放入 `raw/`，转换产物放入 `interim/`，本地短视频放入 `samples/`。这些内容均由 `.gitignore` 排除。
