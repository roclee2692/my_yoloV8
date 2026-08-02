# Data policy and MOT17 setup

The repository does not distribute raw datasets or input videos. 数据、视频及转换产物默认不随仓库分发：

- `raw/`: user-downloaded source datasets such as MOT17; never committed；
- `interim/`: converted videos and validation summaries; never committed；
- `samples/`: local short input media for integration runs; never committed by default。

Only `.gitkeep` placeholders are versioned. The ignore rules cover all contents below these directories, so a normal `git add .` will not stage MOT17 images, GT, converted MP4 files, or generated JSON.

## MOT17 official source / 官方来源

Dataset: **MOT17 Challenge**, from the official [MOTChallenge MOT17 data page](https://motchallenge.net/data/MOT17/).

As checked on 2026-08-02, the official page offers the complete archive (about 5.5 GB), an annotations/detections-only archive, and a development kit. Download the full archive through that official page and extract it locally. This repository intentionally has no automatic downloader because the archive is large and users must review the dataset terms before use.

MOT17 repeats the MOT16 image sequences for the DPM, FRCNN, and SDP public-detection variants, while providing new MOT17 ground truth and detections. For this project, `MOT17-04-SDP` is the documented conversion example; experiments must record the exact variant used.

## License and responsible use / 许可与使用注意

The [MOTChallenge home page](https://motchallenge.net/) publishes its datasets under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)**. Attribute the dataset and cited source sequences, do not use it commercially under those terms, and apply the stated share-alike condition to adaptations. Review the official site again before redistribution or publication; this project’s MIT license does not replace the dataset license.

The data contains identifiable people in public scenes. Keep raw data local, limit access appropriately, and do not use this research benchmark as an identity-recognition dataset.

Recommended citations are listed on the official MOT17 page, including the MOT16 benchmark paper. The MOT16/17 annotation specification is described in [Milan et al., “MOT16: A Benchmark for Multi-Object Tracking”](https://arxiv.org/abs/1603.00831).

## Expected directory structure / 目录结构

After extraction, use this layout:

```text
data/raw/MOT17/
├── train/
│   ├── MOT17-04-SDP/
│   │   ├── img1/
│   │   │   ├── 000001.jpg
│   │   │   └── ...
│   │   ├── gt/
│   │   │   └── gt.txt
│   │   └── seqinfo.ini
│   └── ...
└── test/
    └── MOT17-XX-SDP/
        ├── img1/
        └── seqinfo.ini
```

Public test sequences do not contain Ground Truth. Training-sequence `gt/gt.txt` rows contain nine comma-separated values:

```text
frame, track_id, bbox_left, bbox_top, bbox_width, bbox_height, confidence, class, visibility
```

The Phase 4 reader preserves every row and all nine fields. It does not silently discard ignored entries or non-pedestrian annotation classes; the Phase 7 evaluation policy will apply explicit filters.

## Validate the download / 验证完整性

Run the local validator after extraction:

```bash
python scripts/prepare_mot17.py --root data/raw/MOT17 --split train
python scripts/prepare_mot17.py --root data/raw/MOT17 --split test
```

By default it requires the complete official set of 21 detector-variant directories for the selected split. It also checks every sequence’s `seqinfo.ini`, the complete frame range `1..seqLength`, six-digit frame naming, image extension, and training GT schema. It writes measured totals to:

```text
data/interim/mot17_train_validation.json
data/interim/mot17_test_validation.json
```

A missing frame, extra frame ID, malformed GT row, missing training GT, or inconsistent metadata causes a nonzero exit with a specific path and reason.

For an intentional one-sequence smoke test, add `--allow-subset`. The resulting JSON sets `validation_mode` to `subset` and `official_split_complete`/`complete` to `false`, so subset evidence cannot be mistaken for a complete official download.

## Convert a sequence to MP4 / 转换为视频

```bash
python scripts/mot_sequence_to_video.py \
  --sequence data/raw/MOT17/train/MOT17-04-SDP \
  --output data/interim/MOT17-04-SDP.mp4
```

The converter uses `frameRate`, `imWidth`, and `imHeight` from `seqinfo.ini`, orders images by numeric frame ID, and refuses incomplete inputs. It then decodes the entire generated MP4 again; a frame-count, FPS, or resolution mismatch fails the command and removes the partial file. A successful run creates:

```text
data/interim/MOT17-04-SDP.mp4
data/interim/MOT17-04-SDP.summary.json
```

Existing outputs are preserved unless `--overwrite` is supplied explicitly.

## Phase 3 official sample

The optional local smoke-test video is an Ultralytics official release asset and is not committed:

- name: `People-counting-compressed.mp4`
- source: `https://github.com/ultralytics/assets/releases/download/v0.0.0/People-counting-compressed.mp4`
- size: `358219` bytes
- SHA-256: `e9ebad02081416465caae6ff87a28f96c441337742266b45f29a8c8a7b80c3a5`

Acquire and verify it with `python scripts/download_phase3_assets.py`. CI creates its own synthetic MP4 and never downloads this sample or a model.

This official clip already contains a counting line, boxes, and labels from its original demonstration. It is suitable for repeatable pipeline smoke tests but produces visually overlapping annotations. MOT17 conversion produces clean experiment inputs after the user obtains the official dataset.
