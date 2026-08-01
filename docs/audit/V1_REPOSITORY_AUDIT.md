# V1 Repository Audit

> Project: `roclee2692/my_yoloV8`
> Audit date: 2026-08-01 (Asia/Shanghai)
> Frozen source: `v1.0.0-rps-yolov8` / `6680149accb43191daa33ddb1778172161131d72`
> Audit branch before this report: `v2/yolo26-people-flow` / `e6ca67c47b9b655636b1126a3a2318e484ec3224`

## 1. Executive summary

V1 has been preserved without rewriting `main`: the annotated tag `v1.0.0-rps-yolov8` peels to the original `main` commit, while the V2 documentation commit exists only on `v2/yolo26-people-flow`.

The legacy repository is **not reproducible from a clean machine in its current form**. The main blockers are:

1. `requirements.txt` is referenced by README but is not present, and there is no alternative dependency manifest or lock file.
2. `data.yaml` and three scripts contain machine-specific `D:/Pythonmodel/...` paths.
3. The current audit environment has neither `ultralytics` nor `opencv-python` installed.
4. The repository tracks the dataset, generated runs, extracted results, and five `.pt` files directly in ordinary Git.
5. The dataset metadata says 7,521 images, but the checked-in `train`, `valid`, and `test` splits contain 7,335 images in total.
6. Dataset licensing is not established: the bundled source metadata literally says `License: undefined` and describes aggregation from several other datasets.
7. There are no automated tests. `ruff` finds three issues in the legacy scripts.

No V1 source file was fixed during this phase. All findings describe the frozen repository as it exists.

## 2. Git and repository status

| Item | Observed value |
|---|---|
| Repository | `https://github.com/roclee2692/my_yoloV8` |
| Visibility | Public |
| Default branch | `main` |
| Frozen `main` SHA | `6680149accb43191daa33ddb1778172161131d72` |
| Current V2 branch before audit commit | `v2/yolo26-people-flow` |
| Current V2 SHA before audit commit | `e6ca67c47b9b655636b1126a3a2318e484ec3224` |
| V1 tag | `v1.0.0-rps-yolov8` (annotated) |
| V1 tag peeled commit | `6680149accb43191daa33ddb1778172161131d72` |
| Remote tracking | V2 tracks `origin/v2/yolo26-people-flow`; `main` tracks `origin/main` |
| V1 tracked files | 15,243 |
| V1 tracked bytes | 304,345,724 bytes (about 290.25 MiB, uncompressed) |
| GitHub `diskUsage` | 267,666 KiB (about 261.39 MiB) |
| Local packed objects | 12,319 objects in one 261.72 MiB pack |
| Git LFS | Installed locally, but the repository has no `.gitattributes` and no LFS-tracked files |

Commands used include:

```powershell
git remote -v
git fetch --all --tags --prune
git status --short --branch
git symbolic-ref --short refs/remotes/origin/HEAD
git branch -vv
git tag -n99 --list
git show-ref --tags -d
git log --oneline --decorate --graph --all -10
git count-objects -vH
gh repo view roclee2692/my_yoloV8 --json nameWithOwner,defaultBranchRef,diskUsage,visibility,isPrivate,url
```

### 2.1 Size distribution

The following values were measured from the V2 worktree after adding only the two Phase 0 documents. The legacy asset directory values are unchanged from V1.

| Path | Files | Bytes | Approx. MiB | Classification |
|---|---:|---:|---:|---|
| `datasets/` | 14,674 | 235,713,232 | 224.79 | Dataset and cache |
| `runs/` | 535 | 42,734,458 | 40.75 | Generated training/validation/prediction output |
| `weights/` | 2 | 12,499,028 | 11.92 | Duplicate trained weights |
| `results/` | 23 | 6,840,340 | 6.52 | Extracted copies of run artifacts |
| `yolov8n.pt` | 1 | 6,549,796 | 6.25 | Pretrained weight |

Of the 15,243 files in V1, 15,235 are under `datasets`, `runs`, `results`, `weights`, or are the root `yolov8n.pt` file. Code and documentation are a very small fraction of the repository.

### 2.2 Largest files and thresholds

| File | Bytes | Notes |
|---|---:|---|
| `yolov8n.pt` | 6,549,796 | Root pretrained YOLOv8n weight |
| `runs/detect/train/weights/best.pt` | 6,249,514 | Training output |
| `runs/detect/train/weights/last.pt` | 6,249,514 | Training output |
| `weights/best.pt` | 6,249,514 | Exact copy of run `best.pt` |
| `weights/last.pt` | 6,249,514 | Exact copy of run `last.pt` |
| `datasets/train/labels.cache` | 1,557,009 | Generated Ultralytics cache |

Both the current tree scan and the full-history blob scan found:

- files/blobs over 50 MiB: **0**
- files/blobs over 100 MiB: **0**
- largest historical blob: **6,549,796 bytes** (`yolov8n.pt`)

The full history contained 12,286 blob records. No history rewrite is justified or authorized.

### 2.3 Git LFS decision

Git LFS is **not required to preserve or repair V1**, because no current or historical object exceeds 50 MiB and history must remain untouched. V2 should not use LFS as an excuse to keep committing datasets, generated runs, or ordinary model downloads. The default V2 policy is:

- remove these assets from the V2 worktree in a future approved phase while retaining them in V1 history;
- ignore datasets, generated outputs, caches, and model weights;
- distribute intentionally retained large weights through a model registry, release asset, or separately approved LFS policy rather than ordinary Git.

## 3. Dataset and generated assets

### 3.1 Checked-in dataset

| Split | Images | Label files |
|---|---:|---:|
| `train` | 6,455 | 6,455 |
| `valid` | 576 | 576 |
| `test` | 304 | 304 |
| **Total** | **7,335** | **7,335** |

`datasets/Rock Paper Scissors SXSW.v14i.yolov8/README.roboflow.txt` says the export contains 7,521 images. The working tree contains 7,335 split images, a difference of 186. The audit did not infer whether images were intentionally omitted, filtered, or lost, so dataset completeness is **not established**.

The repository also tracks `datasets/train/labels.cache`, which is a generated cache rather than source data.

### 3.2 Weights

| Path | SHA-256 |
|---|---|
| `yolov8n.pt` | `F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36` |
| `runs/detect/train/weights/best.pt` | `0B7B722CB35BDF73D51D473144E32C9F81549D42E681EAC854C6ECA6B6A30328` |
| `weights/best.pt` | same as run `best.pt` |
| `runs/detect/train/weights/last.pt` | `4434C0D597D0D5AD6845D2FEB1173D788C3B5AAF5C9F9679080E332B19A35B84` |
| `weights/last.pt` | same as run `last.pt` |

The two `weights/` files duplicate the corresponding run artifacts byte-for-byte. None is tracked by Git LFS.

### 3.3 Duplicate and generated files

Grouping the current tree by Git blob ID found 64 duplicate groups accounting for 19,585,285 avoidable bytes (about 18.68 MiB). Important groups include:

- two copies each of `best.pt` and `last.pt`;
- duplicated training plots and validation images in `results/` and `runs/detect/train/`;
- three copies of `results.png`;
- duplicate confusion matrices under `results/`, training output, and validation output;
- duplicate data images and label contents in the dataset.

`runs/detect/` contains `train`, `val`, and `predict` outputs. The `val` directory contains plots and sample batches but no captured `args.yaml`, JSON metrics, or command log tying those artifacts to an exact evaluation invocation.

## 4. Legacy code audit

| File | Observed behavior | Reproducibility and quality issues | Migration decision |
|---|---|---|---|
| `scripts/train.py` | Under a main guard, loads `yolov8n.pt` and trains for 50 epochs at 640 pixels using `data.yaml` | No dependency pin, CLI, explicit output directory, device policy, error handling, or environment capture | `MOVE_TO_LEGACY` |
| `scripts/实时摄像头测试.py` | At import time, loads an absolute-path weight and opens camera 0; uses `show=True`, confidence 0.3 | Import has hardware/model side effects; absolute Windows path; no source/model arguments; no clear camera-open error; result is not persisted | `MOVE_TO_LEGACY` |
| `scripts/推理测试.py` | At import time, loads an absolute-path weight and predicts on an absolute test directory | Absolute paths; unused `os` import; no main guard or CLI; fixed implicit output directory; formal logging absent | `MOVE_TO_LEGACY` |
| `scripts/评估模型.py` | Loads absolute model/data paths and calls `model.val(..., workers=0)` | Windows-specific multiprocessing workaround; no CLI/config schema; result only printed; no version/environment capture | `MOVE_TO_LEGACY` |
| `data.yaml` | Defines Paper, Rock, Scissors classes | All four path fields use `D:/Pythonmodel/my_yolo_project/...` | `MOVE_TO_LEGACY` |
| `README.md` | Describes project, layout, claimed metrics, and limitations | References missing files/paths and unverified claims; insufficient clean-machine instructions | `REWRITE` |
| `.gitignore` | Generic Python ignore template | Ignores `.env` and caches but not datasets, runs, results, weights, `*.pt`, or other model formats | `REWRITE` |
| `LICENSE` | MIT License, copyright 2025 Raelon Veritas Lee | Dependency/model/dataset obligations are not reconciled in project documentation | `KEEP`, then review compatibility for V2 |

### 4.1 Hard-coded paths and model references

The targeted search found nine machine-specific absolute-path occurrences:

- four in `data.yaml`;
- two in `scripts/评估模型.py`;
- two in `scripts/推理测试.py`;
- one in `scripts/实时摄像头测试.py`.

The referenced `D:/Pythonmodel/my_yolo_project`, dataset directory, and trained weight do not exist on the audit machine. A valid weight is present inside this repository, but the affected scripts do not use it.

Other hard-coded behavior includes:

- `yolov8n.pt`, `data.yaml`, 50 epochs, and image size 640 in training;
- webcam index 0 and confidence 0.3 in realtime inference;
- `workers=0` in evaluation;
- Ultralytics default output locations rather than an explicit reproducible run directory.

No bare `except` was found because the scripts contain no exception handling at all. Formal output uses `print` in the inference and evaluation scripts instead of `logging`.

### 4.2 README inconsistencies

- `requirements.txt` is listed but absent.
- The displayed image `results/images/sample_detection.jpg` is absent.
- Highlights name `predict.py`, `realtime_cam.py`, and `evaluate.py`, but the actual files have Chinese names and those English files do not exist.
- The dataset directory name in the tree differs from the actual versioned directory.
- The README says `best.pt` runs at 30+ FPS, but no hardware-specific benchmark log supporting that statement is committed.
- The README says `mAP50 ≈ 0.93`; the archived 50th CSV row records `metrics/mAP50(B)=0.94831`. Neither value was independently rerun in this audit.
- The source metadata says 7,521 images while the checked-in split count is 7,335.

## 5. Dependency and runnability audit

### 5.1 Dependency declaration

No tracked dependency or packaging file exists among:

- `requirements*.txt`
- `pyproject.toml`
- `setup.py` / `setup.cfg`
- `Pipfile`
- `poetry.lock`
- `environment.yml`

Therefore there is no requirements file to install or validate. `pip check` reporting no broken packages only describes the pre-existing environment; it does not validate V1 dependencies.

### 5.2 Audit environment

| Component | Observed state |
|---|---|
| Python | 3.13.12 |
| pip | 26.0.1 |
| PyTorch | 2.10.0+cpu |
| Ultralytics | not installed |
| OpenCV (`opencv-python`) | not installed |
| pytest | 9.1.1, installed during Phase 0 validation |
| ruff | 0.16.1, installed during Phase 0 validation |

`python -m compileall -q scripts` succeeds, so the four source files are syntactically compilable. This does not prove runtime success.

The scripts were deliberately **not executed or imported as applications** because doing so would trigger training, model loading, an external absolute path, output generation, or camera access. Based on direct dependency and path checks:

- `train.py` cannot run in this environment because Ultralytics is absent and dependencies are undeclared;
- realtime inference cannot run because Ultralytics and OpenCV are absent and its absolute weight path is missing;
- batch inference cannot run because Ultralytics and both absolute input/model paths are missing;
- evaluation cannot run because Ultralytics and both absolute model/data paths are missing.

### 5.3 Archived training provenance

`runs/detect/train/args.yaml` captures useful partial configuration: YOLOv8n, 50 epochs, batch 16, image size 640, `cuda:0`, 8 workers, AMP enabled, seed 0, and deterministic mode. However:

- it does not record the Ultralytics, Python, PyTorch, CUDA, driver, or GPU versions;
- it names `save_dir: runs\detect\train4`, while the committed artifact is stored under `runs/detect/train`;
- `data.yaml` depends on an unavailable absolute path;
- there is no environment lock or complete command log.

The final CSV row and plots are archived evidence only. They were not rerun and must not be presented as newly verified performance.

## 6. License and provenance audit

This section records source statements and is not legal advice.

### 6.1 Project code

The root `LICENSE` is MIT and names Raelon Veritas Lee as the 2025 copyright holder.

### 6.2 Dataset

The bundled `README.dataset.txt` links to the official Roboflow Universe page and explicitly states `License: undefined`. It also says the dataset aggregates material from multiple Roboflow datasets and adds new images. No separate license files for those component datasets are committed.

Sources checked on 2026-08-01:

- Local metadata: `datasets/Rock Paper Scissors SXSW.v14i.yolov8/README.dataset.txt`
- Official dataset page: <https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw>

Conclusion: redistribution rights for the checked-in dataset are unresolved. Do not carry the image/label corpus into the V2 working tree or future releases until the source licenses are documented.

### 6.3 Ultralytics and model weights

The current official Ultralytics repository states that its open-source code is AGPL-3.0 and describes an Enterprise License option for uses that do not follow the AGPL requirements:

- <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>
- <https://github.com/ultralytics/ultralytics/blob/main/docs/README.md>

V1 does not document the exact Ultralytics package version, the provenance/download record of `yolov8n.pt`, or the intended license treatment for the fine-tuned weights. V2 must document the selected Ultralytics version and applicable license before distribution or deployment.

## 7. Security and repository hygiene

A path-only, redacted-content scan checked tracked text for common GitHub tokens, OpenAI-style keys, AWS access keys, and assignments named API key, secret, token, or password. It found no matching tracked files. No `.env`, PEM, key, credentials, or secrets filename is tracked.

This lightweight scan reduces obvious risk but is not a substitute for GitHub secret scanning or a dedicated tool such as Gitleaks.

Current hygiene issues:

- datasets and generated artifacts are committed rather than ignored;
- ordinary Git contains all five model weights;
- caches and duplicated reports are committed;
- dependency versions and runtime environment are absent;
- there are no CI workflows or automated tests;
- logs/results are not tied to immutable experiment configurations.

## 8. Duplicate, obsolete, and non-source content

The following should not remain as V2 source-controlled runtime content:

- `datasets/train/**`, `datasets/valid/**`, `datasets/test/**`;
- `datasets/train/labels.cache`;
- `runs/**`;
- `results/**`;
- `weights/**` and root `yolov8n.pt`;
- future `.pt`, `.pth`, `.onnx`, `.engine`, OpenVINO, TensorRT, and similar model artifacts;
- local virtual environments, caches, logs, secrets, uploaded data, and generated videos/CSV/JSON runs.

Removal in a later V2 phase will affect only the V2 tree. V1 remains recoverable from the tag and Git history; no history rewrite is planned.

## 9. Complete migration decision table

The rules below cover every V1 path either individually or by directory pattern. Phase 1 records these decisions but does not execute them.

| V1 path or group | Decision | V2 treatment |
|---|---|---|
| `LICENSE` | `KEEP` | Retain provisionally; document dependency/model license compatibility |
| `README.md` | `REWRITE` | Replace with the reproducible people-flow project documentation |
| `.gitignore` | `REWRITE` | Add data, outputs, weights, model exports, caches, environments, secrets, logs, and dashboard uploads |
| `data.yaml` | `MOVE_TO_LEGACY` | Preserve under `legacy/rps_yolov8/`; new V2 configuration is unrelated |
| `scripts/train.py` | `MOVE_TO_LEGACY` | Preserve as historical RPS training code |
| `scripts/实时摄像头测试.py` | `MOVE_TO_LEGACY` | Preserve as historical webcam code |
| `scripts/推理测试.py` | `MOVE_TO_LEGACY` | Preserve as historical batch-inference code |
| `scripts/评估模型.py` | `MOVE_TO_LEGACY` | Preserve as historical evaluation code |
| `datasets/Rock Paper Scissors SXSW.v14i.yolov8/README*.txt` | `MOVE_TO_LEGACY` | Preserve provenance metadata only |
| `datasets/train/**`, `datasets/valid/**`, `datasets/test/**` | `REMOVE_FROM_V2` | Keep only in V1 tag/history; do not migrate the corpus |
| `runs/**` | `REMOVE_FROM_V2` | Keep only in V1 tag/history |
| `results/**` | `REMOVE_FROM_V2` | Keep only in V1 tag/history |
| `weights/**` and `yolov8n.pt` | `REMOVE_FROM_V2` | Keep only in V1 tag/history; future weights are external artifacts |
| Future raw/interim datasets, runs, videos, CSV/JSON outputs, weights, caches, local envs, secrets, and logs | `DO_NOT_COMMIT` | Add explicit V2 ignore rules and documentation |

## 10. Validation evidence

Commands executed without changing the V1 tag include:

```powershell
rg -n -S "D:/|D:\\\\|Pythonmodel" scripts data.yaml README.md
rg -n -S "best\.pt|yolov8n\.pt|data\.yaml|datasets" scripts data.yaml README.md
Get-ChildItem -Recurse -File | Sort-Object Length -Descending
git ls-tree -r -l HEAD
git rev-list --objects --all | git cat-file --batch-check=...
git lfs version
git lfs ls-files
python --version
python -m pip --version
python -m pip show torch
python -m pip show ultralytics
python -m pip show opencv-python
python -m pip check
python -m compileall -q scripts
pytest -q
ruff check .
```

Observed checks before the audit commit:

| Check | Result |
|---|---|
| `python -m compileall -q scripts` | Passed; no syntax error |
| `python -m pip check` | `No broken requirements found` in the ambient environment |
| `pytest -q` | Non-zero; `no tests ran` (0 passed, 0 failed, 0 skipped because no tests were collected) |
| `ruff check .` | Non-zero; 3 findings: two import-order findings and one unused `os` import |

The final Phase 1 validation reran `pytest -q` and `ruff check .` after this report was written; the results were unchanged. Failures remain documented rather than fixed because Phase 1 is audit-only.

## 11. Phase 1 conclusion

The V1 preservation is sound, but the legacy working tree is unsuitable as the foundation of the people-flow system. V2 should begin from a clean `src`-layout skeleton while keeping a small, clearly labeled legacy code/provenance area. Dataset images, generated runs, result copies, and weights should leave the V2 tree without rewriting V1 history.

No V2 restructuring, YOLO26 download, MOT17 download, tracking implementation, training, dashboard, or LLM report was performed in this audit.
