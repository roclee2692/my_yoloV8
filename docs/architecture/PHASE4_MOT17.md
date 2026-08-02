# Phase 4 MOT17 data architecture

## Scope

Phase 4 adds input and Ground Truth support only. It does not implement line crossing, occupancy, tracking evaluation, model training, or experiment comparisons.

## Data flow

```text
data/raw/MOT17/<split>/<sequence>/seqinfo.ini
                         │
                         ├── img1/000001.jpg ...  ──> strict frame-set validation
                         │                                  │
                         └── gt/gt.txt ──> 9-column parser   ├──> MP4 converter
                                              │             │       │
                                              └──> frame GT │       └──> full decode verification
                                                            │
                                                            └──> JSON integrity summaries
```

## Responsibilities

- `mot_reader.py` parses `seqinfo.ini`, validates paths and scalar metadata, proves that the numeric frame set exactly equals `1..seqLength`, and reads all MOT16/17 GT fields.
- `mot_ground_truth.py` groups the raw, unfiltered records by frame without introducing evaluation policy.
- `mot_converter.py` writes the images in numeric order at the declared FPS and size. It uses a temporary MP4 and only promotes it to the requested output after decoding every frame and checking FPS, size, and count.
- `prepare_mot17.py` requires all 21 official detector-variant directories by default, validates their contents, and writes measured totals. Explicit subset mode never marks the official split complete. The script does not download or alter raw data.
- `mot_sequence_to_video.py` is a thin CLI around the reusable converter.

## Invariants

1. Frame IDs are one-based, six-digit names and cover the entire declared range.
2. A training sequence must contain `gt/gt.txt`; a public test sequence may omit it.
3. Ground Truth rows contain exactly nine finite fields. Frame, trajectory, and class IDs are integral; visibility is in `[0, 1]`; boxes have positive size.
4. Raw GT is not silently filtered. Person-class and ignored-region policy belongs to Phase 7 so predicted and Ground Truth counting can share one explicit algorithm.
5. Existing MP4/JSON outputs require `--overwrite`; a failed conversion never leaves a partial video under the requested final name.
6. Raw data, converted media, and generated validation summaries remain ignored by Git.
7. `complete: true` requires all 21 official sequence names for the selected split; a deliberate subset is labeled as incomplete even when its own contents validate.

## Source basis

- [MOT17 official data page](https://motchallenge.net/data/MOT17/)
- [MOTChallenge license statement](https://motchallenge.net/)
- [MOT16/17 annotation specification](https://arxiv.org/abs/1603.00831)
- [Official MOTChallenge evaluation format](https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md)

The MOTChallenge website states CC BY-NC-SA 3.0 for its datasets. Repository code licensing does not override those dataset terms.
