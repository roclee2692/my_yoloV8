# Phase 10 Streamlit Dashboard

## Scope

Phase 10 adds a local Streamlit web interface after the core detection, tracking,
counting, Ground Truth evaluation, experiment matrix, and training-readiness gate.
It does not introduce another copy of any computer-vision or counting algorithm.

The Dashboard supports:

- selecting a repository sample or uploading an MP4;
- selecting YOLO26n/YOLOv8n, ByteTrack/BoT-SORT, confidence, and device;
- clicking the source video's first frame to define a directed line and polygon ROI;
- starting the canonical pipeline with visible frame progress;
- playing `annotated.mp4`;
- reading current occupancy, directional totals, peak occupancy/time, and runtime FPS;
- plotting occupancy/cumulative flow and dwell-time distribution;
- inspecting the Phase 8 experiment comparison;
- downloading a run ZIP or the comparison CSV.

## Architecture and trust boundary

```text
Streamlit widgets
    -> DashboardRunRequest
    -> canonical RunConfig validation
    -> people_flow.pipeline.run_pipeline
    -> CSV / JSON / annotated MP4 artifacts
    -> validated DashboardRunData
    -> KPI cards, charts, tables, video, downloads
```

`src/people_flow/dashboard/service.py` is a thin input and orchestration adapter.
It uses `DirectedLine` and `PolygonRegion` for geometry validation and delegates the
job to `run_pipeline`. `src/people_flow/dashboard/data.py` performs read-only,
cross-artifact validation before values are rendered. The Streamlit module does not
detect, track, count, or recompute evaluation metrics.

Uploaded filenames are reduced to a basename, sanitized, and combined with a SHA-256
content prefix. Uploads remain under ignored `data/uploads/`. Run IDs accept only
portable characters, so they cannot escape `runs/dashboard/`.

## Ground Truth policy

The Dashboard never converts an unavailable accuracy value into zero. If the selected
run does not have `evaluation.json` with `ground_truth_available: true`, the UI shows:

> 本次结果没有人工或数据集 Ground Truth 验证，仅为模型预测结果。

Phase 8 rows without Ground Truth also show a visible warning. FPS can still be
compared, but counting accuracy and ID Switch conclusions remain unavailable.

## Installation and launch

```powershell
cd F:\my_yoloV8
.venv\Scripts\python.exe -m pip install -e ".[dev,dashboard]"
$env:PEOPLE_FLOW_PROJECT_ROOT = (Resolve-Path .).Path
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Open `http://localhost:8501`. The Dashboard uses the CUDA device option only when
the Python interpreter running Streamlit reports `torch.cuda.is_available() == True`.
To use the user's `ai_env` GPU environment, install this repository and the Dashboard
extra into that environment, then launch Streamlit with that interpreter.

## Verification

The automated suite covers:

- safe upload persistence and portable run IDs;
- source/display coordinate mapping and first-frame decoding;
- canonical `RunConfig` construction and one-way pipeline delegation;
- initial and per-frame pipeline progress updates;
- run artifact discovery, schema reconciliation, derived chart series, and ZIP export;
- a model-free Streamlit `AppTest` render with KPI cards and the no-GT warning.

Real browser acceptance must additionally inspect the rendered local page, tabs,
controls, video/result panel, chart canvases, and Ground Truth warning. CI installs the
Dashboard extra but does not download weights, start inference, or require a GPU.

## Known limitations

- The drawing surface uses click-to-add vertices rather than drag handles.
- Processing runs in the Streamlit server process; it is suitable for a local,
  single-operator demonstration, not a multi-tenant job queue.
- Existing Phase 8 sample results have no Ground Truth, so the Dashboard cannot prove
  model/tracker accuracy differences from them.
- Authentication, persistent job scheduling, and remote deployment are outside Phase 10.
