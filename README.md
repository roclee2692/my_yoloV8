# People Flow Analytics with YOLO26 and Multi-Object Tracking

基于 YOLO26 与多目标跟踪的固定摄像头人流分析系统。

## 中文

### 当前状态：V2 Phase 3

本分支已经建立可安装、可测试的 `src` layout，并实现最小可运行的人体检测与多目标跟踪管线：

- 严格校验的基础 YAML 配置；
- `people-flow` 命令行入口、配置检查和 `run` 命令；
- YOLOv8n/YOLO26n 官方权重的校验下载及明确失败处理；
- ByteTrack/BoT-SORT 切换、person-only 过滤和持久 Track ID；
- 标注视频、轨迹 CSV、运行配置、运行日志和性能 JSON；
- 无 GPU、无模型下载的 CI 单元与微型视频集成测试；
- V1 历史代码与数据来源说明的 `legacy/` 归档。

虚拟线计数、ROI、停留时间和 Ground Truth 评测仍按后续阶段实施。README 不填写未经真实实验得到的性能指标。

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
pip install -e ".[dev]"
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
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

输出包括：

```text
runs/demo/
├── annotated.mp4
├── tracks.csv
├── run_config.yaml
├── runtime_metrics.json
└── run.log
```

### 质量检查

```bash
ruff check .
pytest -q
python -m mypy src tests
```

CI 不下载大型模型，也不依赖 GPU。

### 数据和权重

- `data/raw/`、`data/interim/`、`data/samples/` 只保留目录占位文件；
- 原始 MOT17、MOT20、CrowdHuman 数据不会随仓库分发；
- `.pt`、ONNX、TensorRT 等模型文件默认不提交普通 Git；
- 所有运行输出写入已忽略的 `runs/` 子目录。

更详细的数据策略见 `data/README.md`。

## English

### Status: V2 Phase 3

This branch contains an installable `src`-layout application and the minimal person detection and multi-object tracking pipeline. It supports verified official YOLO26n/YOLOv8n acquisition, ByteTrack/BoT-SORT, person-only records, annotated MP4 output, CSV trajectories, effective configuration capture, logs, and runtime metrics.

Counting, ROI occupancy, dwell time, and Ground Truth evaluation remain intentionally deferred. No unverified performance result is claimed.

### Quick start

```bash
python -m venv .venv
pip install -e ".[dev]"
people-flow validate-config --config configs/base.yaml
python scripts/download_phase3_assets.py
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device cpu --output-dir runs/demo
ruff check .
pytest -q
python -m mypy src tests
```

The frozen Rock-Paper-Scissors V1 is available at `v1.0.0-rps-yolov8`. Dataset files, generated runs, and weights remain recoverable from that tag but are excluded from the V2 tree.

## License

The repository currently retains its MIT project license. Ultralytics code and models are offered under AGPL-3.0 and an Enterprise license; deploying the future web application must follow the selected Ultralytics license. The repository license does not override dependency, model, video, or dataset terms. See the official [Ultralytics licensing page](https://www.ultralytics.com/license) and the V1 audit in `docs/audit/V1_REPOSITORY_AUDIT.md`.
