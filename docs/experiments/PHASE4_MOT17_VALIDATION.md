# Phase 4 MOT17 validation record

Date: 2026-08-02
Branch: `v2/yolo26-people-flow`

## Evidence boundary

No official MOT17 archive was present at `data/raw/MOT17` during this phase, and the 5.5 GB dataset was not downloaded. Therefore this record does **not** claim a successful run on a real MOT17 sequence.

The implementation was verified with a generated three-frame fixture that exactly follows the relevant MOT17 layout and schema:

```text
MOT17/train/MOT17-99-SDP/
├── img1/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── 000003.jpg
├── gt/
│   └── gt.txt
└── seqinfo.ini
```

Fixture metadata: 3 frames, 7 FPS, 32x24 pixels, `.jpg`; GT contained three valid nine-column observations. The fixture and outputs live below ignored `data/interim/` and are not versioned.

## Commands executed

```powershell
.venv\Scripts\python.exe scripts/prepare_mot17.py `
  --root data/interim/phase4_cli_validation/MOT17 `
  --split train `
  --allow-subset `
  --output data/interim/phase4_cli_validation/validation.json

.venv\Scripts\python.exe scripts/mot_sequence_to_video.py `
  --sequence data/interim/phase4_cli_validation/MOT17/train/MOT17-99-SDP `
  --output data/interim/phase4_cli_validation/MOT17-99-SDP.mp4
```

## Measured outputs

`validation.json` reported:

```json
{
  "validation_mode": "subset",
  "sequence_count": 1,
  "total_frames": 3,
  "total_ground_truth_records": 3,
  "content_validated": true,
  "official_split_complete": false,
  "complete": false
}
```

`MOT17-99-SDP.summary.json` reported:

```json
{
  "input_frames": 3,
  "output_frames": 3,
  "frame_rate": 7.0,
  "width": 32,
  "height": 24,
  "codec": "mp4v",
  "output_size_bytes": 938,
  "frame_count_verified": true
}
```

An independent pass through the project `VideoSource` decoded exactly 3 frames at 7.0 FPS and 32x24.

## Negative-path evidence

The command below was also run against the documented default path:

```powershell
.venv\Scripts\python.exe scripts/prepare_mot17.py `
  --root data/raw/MOT17 `
  --split train
```

It returned exit code `2` with:

```text
MOT17 train directory does not exist: F:\my_yoloV8\data\raw\MOT17\train.
Download and extract MOT17 according to data/README.md.
```

This is the expected truthful result while the official dataset is absent.

Running the synthetic one-sequence root without `--allow-subset` also returned exit code `2`; the error listed the 21 missing official train names and the synthetic extra name. This proves that subset validation cannot silently claim a complete MOT17 split.

## Automated coverage

The Phase 4 tests cover numeric image ordering, metadata parsing, all nine GT fields, frame-indexed GT access, missing frames, malformed GT, public-test GT absence, split summaries, round-trip video parity, explicit overwrite, and frame-size mismatch cleanup. They create no model downloads and require neither GPU nor network access.

## Remaining real-data acceptance

After the user obtains MOT17 from the [official data page](https://motchallenge.net/data/MOT17/), run the documented validation command and convert `MOT17-04-SDP`. Those future outputs must replace neither this synthetic evidence nor the distinction that real-dataset validation is still pending.
