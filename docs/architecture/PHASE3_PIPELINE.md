# Phase 3 detection and tracking pipeline

The command line interface builds a validated `RunConfig` and calls the same `run_pipeline` API intended for the future web interface.

```text
MP4 file
  -> VideoSource (frame index, timestamp, source FPS)
  -> YoloDetector (exact requested weights, person class 0)
  -> TrackerRunner (ByteTrack or BoT-SORT, persistent IDs)
  -> TrackRecord (bbox, center, bottom-center foot point)
  -> CSV writer + frame annotator + MP4 writer
  -> runtime_metrics.json + run_config.yaml + run.log
```

The official model registry stores the release URL, byte length, and SHA-256 for `yolo26n.pt` and `yolov8n.pt`. A missing known model is downloaded to `weights/`, verified, and never committed. Unknown model names and unavailable CUDA requests fail explicitly; there is no fallback model or device.

CI exercises the complete orchestration with a generated three-frame MP4 and an injected fake tracker. It verifies stable IDs, an empty-detection frame, required artifacts, frame count, and source FPS without downloading weights. Real model execution is a separate local acceptance check and its measured outputs remain under ignored `runs/`.
