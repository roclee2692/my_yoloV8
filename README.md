# People Flow Analytics with YOLO26 and Multi-Object Tracking

基于 YOLO26 与多目标跟踪的固定摄像头人流分析系统。

## 中文

### 当前状态：V2 Phase 11

本分支已经建立可安装、可测试的 `src` layout，并实现最小可运行的人体检测与多目标跟踪管线：

- 严格校验的基础 YAML 配置；
- `people-flow` 命令行入口、配置检查和 `run` 命令；
- YOLOv8n/YOLO26n 官方权重的校验下载及明确失败处理；
- ByteTrack/BoT-SORT 切换、person-only 过滤和持久 Track ID；
- 标注视频、轨迹 CSV、运行配置、运行日志和性能 JSON；
- 无 GPU、无模型下载的 CI 单元与微型视频集成测试；
- MOT17 `seqinfo.ini`、完整图像序列与九列 Ground Truth 的严格读取；
- MOT17 train/test 本地完整性校验及 JSON 摘要；
- 按源 FPS 和分辨率转换 MP4，并逐帧回读证明输入/输出帧数一致；
- 基于 Track ID 与脚点轨迹的有向虚拟线进出计数；
- 最小轨迹年龄、最小位移、冷却帧、短时丢失恢复和同向去重；
- 逐事件 `events.csv` 与 JPEG 证据帧；
- 基于脚点和 Track ID 的多边形 ROI 占用、进入与离开状态；
- 首次进入、最后出现、累计停留时间及 `occupancy.csv`、`dwell_times.csv`、`summary.json`；
- 将 MOT Ground Truth 与预测 CSV 统一转换为 `TrackRecord`；
- 使用完全相同的虚拟线、脚点、冷却和 ROI 算法生成 GT 与预测事件；
- 自动计算进出绝对误差、总计数绝对误差、count MAE、零安全 MAPE 与 occupancy MAE；
- 严格 YAML 驱动的 YOLOv8n/YOLO26n 与 ByteTrack/BoT-SORT A/B/C 实验矩阵；
- 每实验有效配置、预测与运行指标、显式 GT 可用性及聚合 `comparison.csv`；
- 基于一对一 IoU 关联的透明 ID Switch 指标（有 MOT Ground Truth 时启用）；
- 基于真实基线证据的 Phase 9 训练准入审计与结构化延期决策；
- 调用核心 API 的 Streamlit Dashboard、可视化几何配置、进度、视频、KPI、图表与结果下载；
- 仅使用三个结构化 JSON 的防编造 LLM 报告管线、Markdown 报告与 Dashboard 下载；
- V1 历史代码与数据来源说明的 `legacy/` 归档。

Phase 8 已在本地 RTX 4060 上完成真实 A/B/C 功能与运行性能基线。当前样例视频没有 Ground Truth，因此 ID Switch 与所有准确率误差均明确为 `null`。Phase 9 自动准入审计据此延期训练，没有生成伪造权重、曲线或指标。Phase 10 已增加本地 Web Dashboard；页面读取真实 CSV/JSON 产物，并持续显示 Ground Truth 可用性。Phase 11 已增加结构化报告管线；外部 LLM 仅生成无数字叙述，所有指标由经过校验的 JSON 确定性写入。

### V1 历史版本

YOLOv8 石头剪刀布项目冻结在标签：

```text
v1.0.0-rps-yolov8
```

V1 的数据集、训练产物和权重仍保存在该标签与 Git 历史中，但不再出现在 V2 工作树。旧脚本和数据来源说明位于 `legacy/rps_yolov8/`。

### 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,dashboard]"
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,dashboard]"
```

### 当前可用命令

```bash
people-flow --help
people-flow validate-config --config configs/base.yaml
python -m people_flow.cli validate-config --config configs/base.yaml
```

下载并校验官方 YOLO26n 权重和小型人流样例：

```bash
python scripts/download_phase3_assets.py
```

使用 CPU + YOLO26n + ByteTrack：

```bash
people-flow run \
  --source data/samples/People-counting-compressed.mp4 \
  --model yolo26n.pt \
  --tracker bytetrack.yaml \
  --classes 0 \
  --device cpu \
  --output-dir runs/demo
```

也可以切换为 `yolov8n.pt` 或 `botsort.yaml`。模型名称不会静默回退；自定义模型必须提供真实存在的路径。重复使用已有输出目录时需明确增加 `--overwrite`。

启用有向虚拟线计数：

```bash
people-flow run \
  --source data/samples/People-counting-compressed.mp4 \
  --model yolo26n.pt \
  --tracker bytetrack.yaml \
  --classes 0 \
  --device 0 \
  --output-dir runs/counting_demo \
  --counting-line 100 500 1100 500 \
  --enter-side positive \
  --min-track-age 5 \
  --min-displacement-pixels 15 \
  --cooldown-frames 30 \
  --max-track-gap-frames 30
```

`p1 -> p2` 定义有向有限线段。坐标位于有向线左侧时为 `positive`，右侧时为 `negative`；交换 `--enter-side` 即可交换 enter/exit 定义。只有脚点运动线段实际穿过有限计数线、满足轨迹年龄与位移阈值时才产生事件。
启用多边形 ROI 占用与停留时间：

```bash
people-flow run \
  --source data/samples/People-counting-compressed.mp4 \
  --model yolo26n.pt \
  --tracker bytetrack.yaml \
  --classes 0 \
  --device 0 \
  --output-dir runs/roi_demo \
  --roi-name entrance_area \
  --roi-point 100 100 \
  --roi-point 1100 100 \
  --roi-point 1100 650 \
  --roi-point 100 650 \
  --roi-max-track-gap-frames 30
```

每个 `--roi-point X Y` 按顺序定义一个顶点，至少需要三个点。多边形边界视为 ROI 内部；占用人数只统计当前帧真实观测到的脚点。短暂 Track 丢失可延续同一次停留，但不会被写入当前帧占用人数。

输出包括：

```text
runs/demo/
├── annotated.mp4
├── tracks.csv
├── events.csv                 # 仅启用计数时生成
├── evidence/                  # 每个计数事件的证据帧
├── occupancy.csv              # 仅启用 ROI 时生成，每帧一行
├── dwell_times.csv            # 每个进入过 ROI 的 Track 一行
├── summary.json               # ROI 占用与停留汇总
├── run_config.yaml
├── runtime_metrics.json
└── run.log
```

### MOT17 数据

完整 MOT17 数据集不会随仓库分发。通过 MOTChallenge 官方页面获取并解压到 `data/raw/MOT17/`，然后运行：

```bash
python scripts/prepare_mot17.py --root data/raw/MOT17 --split train
python scripts/mot_sequence_to_video.py \
  --sequence data/raw/MOT17/train/MOT17-04-SDP \
  --output data/interim/MOT17-04-SDP.mp4
```

转换成功后会同时生成 `MOT17-04-SDP.summary.json`。脚本严格使用 `seqinfo.ini` 的 FPS、分辨率和帧数，并回读完整输出验证无丢帧。数据目录、官方链接、许可说明及验证规则见 `data/README.md`。

使用同一套生产计数算法评测预测轨迹：

```bash
python scripts/evaluate_experiment.py \
  --mot-sequence data/raw/MOT17/train/MOT17-04-SDP \
  --predicted-tracks runs/MOT17-04-SDP/tracks.csv \
  --output-dir runs/MOT17-04-SDP/evaluation \
  --counting-line 100 500 1100 500 \
  --enter-side positive \
  --min-track-age 5 \
  --min-displacement-pixels 15 \
  --cooldown-frames 30 \
  --roi-name entrance_area \
  --roi-point 100 100 \
  --roi-point 1100 100 \
  --roi-point 1100 650 \
  --roi-point 100 650
```

默认只使用 MOT class 1、marked 且满足 visibility 阈值的行人标注。输出包含 `ground_truth_events.csv`、`predicted_events.csv`、对应 ROI 文件与 `evaluation.json`。当 Ground Truth 总穿越数为 0 时，`count_mape` 为 `null`，不会除以零。完整定义见 `docs/architecture/PHASE7_GROUND_TRUTH_EVALUATION.md`。

### Phase 8 基线实验

使用用户已配置且支持 CUDA 的 Python 环境运行固定矩阵：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
& "C:\Users\<USER>\miniconda3\envs\ai_env\python.exe" `
  scripts\run_baseline.py `
  --config configs\experiments\baseline.yaml `
  --project-root .
```

三组实验只改变模型或跟踪器，其他输入、阈值、几何、设备和 seed 完全共享。输出位于 `runs/experiments/`；详细设计和本次真实结果见 `docs/architecture/PHASE8_EXPERIMENT_MATRIX.md` 与 `docs/experiments/BASELINE_RESULTS.md`。样例没有 Ground Truth 时，`evaluation.json` 不会把未知误差伪装成零。

### Phase 9 训练准入

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.venv\Scripts\python.exe scripts\assess_training_readiness.py `
  --comparison runs\experiments\comparison.csv `
  --output runs\training\phase9_readiness.json
```

当前实际结果为 `decision: deferred`、`training_allowed: false`。只有在 MOT17 Ground Truth、检测 Recall 和计数误差归因证明检测质量是主要瓶颈后，才允许启动 YOLO26n 微调。完整决策见 `docs/experiments/PHASE9_TRAINING_DECISION.md`。

### Phase 10 Dashboard

```powershell
$env:PEOPLE_FLOW_PROJECT_ROOT = (Resolve-Path .).Path
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

浏览器打开 `http://localhost:8501`。界面支持 MP4 上传、模型/跟踪器/置信度/设备选择、首帧点击绘制计数线与 ROI、处理进度、标注视频、KPI、人流与停留时间图表、实验对比及结果下载。Dashboard 只调用核心 Python API；没有 Ground Truth 时会显示明确警告。完整设计见 `docs/architecture/PHASE10_DASHBOARD.md`。

### Phase 11 结构化 LLM 报告

离线、无密钥的可复现模式：

```powershell
people-flow report `
  --run-dir runs\experiments\C_yolo26n_botsort `
  --provider deterministic
```

输出为同一运行目录下的 `report.md` 与 `report_metadata.json`。确定性模式会明确记录 `llm_used: false`，不会冒充外部模型调用。若使用显式配置的 OpenAI-compatible endpoint，API Key 只从 `--api-key-env` 指定的环境变量读取。LLM 只能返回不含数字的解释文本；视频、轨迹 CSV 和证据帧不会进入提示词，精确指标全部由结构化数据渲染。完整设计见 `docs/architecture/PHASE11_LLM_REPORT.md`。

### 质量检查

```bash
ruff check .
pytest -q
python -m mypy src tests scripts
```

CI 不下载大型模型，也不依赖 GPU。

### 数据和权重

- `data/raw/`、`data/interim/`、`data/samples/` 只保留目录占位文件；
- 原始 MOT17、MOT20、CrowdHuman 数据不会随仓库分发；
- `.pt`、ONNX、TensorRT 等模型文件默认不提交普通 Git；
- 所有运行输出写入已忽略的 `runs/` 子目录。

更详细的数据策略见 `data/README.md`。

## English

### Status: V2 Phase 11

This branch contains an installable `src`-layout application, person detection and tracking, strict MOT17 support, directional line crossing, polygon ROI analytics, shared-algorithm Ground Truth evaluation, and a strict A/B/C model-tracker experiment matrix with per-run and aggregate artifacts.

A real RTX 4060 A/B/C runtime baseline has been completed on the bundled sample. Because it has no Ground Truth, the Phase 9 readiness gate deferred training. No training run, weight, curve, or validation metric was fabricated. Phase 10 adds a local Streamlit Dashboard that calls the core API and renders persisted artifacts with an explicit Ground Truth caveat. Phase 11 adds a structured report pipeline in which optional LLM prose cannot introduce numeric claims; exact values come only from validated JSON evidence.

### Quick start

```bash
python -m venv .venv
pip install -e ".[dev,dashboard]"
people-flow validate-config --config configs/base.yaml
python scripts/download_phase3_assets.py
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device cpu --output-dir runs/demo
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device 0 --output-dir runs/counting_demo --counting-line 100 500 1100 500 --enter-side positive
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device 0 --output-dir runs/roi_demo --roi-name entrance_area --roi-point 100 100 --roi-point 1100 100 --roi-point 1100 650 --roi-point 100 650
python scripts/prepare_mot17.py --root data/raw/MOT17 --split train
python scripts/mot_sequence_to_video.py --sequence data/raw/MOT17/train/MOT17-04-SDP --output data/interim/MOT17-04-SDP.mp4
python scripts/evaluate_experiment.py --mot-sequence data/raw/MOT17/train/MOT17-04-SDP --predicted-tracks runs/MOT17-04-SDP/tracks.csv --output-dir runs/MOT17-04-SDP/evaluation --counting-line 100 500 1100 500 --roi-point 100 100 --roi-point 1100 100 --roi-point 1100 650 --roi-point 100 650
python scripts/run_baseline.py --config configs/experiments/baseline.yaml --project-root .
python scripts/assess_training_readiness.py --comparison runs/experiments/comparison.csv --output runs/training/phase9_readiness.json
python -m streamlit run dashboard/app.py
people-flow report --run-dir runs/experiments/C_yolo26n_botsort --provider deterministic
ruff check .
pytest -q
python -m mypy src tests scripts
```

The frozen Rock-Paper-Scissors V1 is available at `v1.0.0-rps-yolov8`. Dataset files, generated runs, and weights remain recoverable from that tag but are excluded from the V2 tree.

## License

The repository currently retains its MIT project license. Ultralytics code and models are offered under AGPL-3.0 and an Enterprise license; deploying the future web application must follow the selected Ultralytics license. The repository license does not override dependency, model, video, or dataset terms. See the official [Ultralytics licensing page](https://www.ultralytics.com/license) and the V1 audit in `docs/audit/V1_REPOSITORY_AUDIT.md`.
