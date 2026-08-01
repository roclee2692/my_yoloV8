# People Flow Analytics with YOLO26 and Multi-Object Tracking

基于 YOLO26 与多目标跟踪的固定摄像头人流分析系统。

## 中文

### 当前状态：V2 Phase 2

本分支已经建立可安装、可测试的 `src` layout 工程骨架。当前阶段只包含：

- 严格校验的基础 YAML 配置；
- `people-flow` 命令行入口及配置检查命令；
- 检测、跟踪、计数、数据集、评测、输出和可视化模块边界；
- 无 GPU、无模型下载的单元测试和 GitHub Actions；
- V1 历史代码与数据来源说明的 `legacy/` 归档。

检测、跟踪、视频输出和短视频集成管线将在 Phase 3 实现。当前没有下载或运行 YOLO26，也没有填写任何未经真实实验得到的指标。

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

`people-flow run` 将在 Phase 3 加入；现在调用未来阶段脚本会返回明确的“尚未实现”错误，而不会下载模型或伪造结果。

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

### Status: V2 Phase 2

This branch now contains an installable and testable `src`-layout scaffold. It provides strict YAML configuration validation, a configuration-checking CLI, package boundaries, model-free tests, CI, and a clearly separated V1 legacy archive.

Detection, tracking, annotated video output, and the short-video pipeline are intentionally deferred to Phase 3. No YOLO26 model was downloaded or executed, and no performance result is claimed.

### Quick start

```bash
python -m venv .venv
pip install -e ".[dev]"
people-flow validate-config --config configs/base.yaml
ruff check .
pytest -q
python -m mypy src tests
```

The frozen Rock-Paper-Scissors V1 is available at `v1.0.0-rps-yolov8`. Dataset files, generated runs, and weights remain recoverable from that tag but are excluded from the V2 tree.

## License

The repository currently retains its MIT project license. Dependency, model, and dataset licenses must also be followed; see the V1 audit in `docs/audit/V1_REPOSITORY_AUDIT.md`.
