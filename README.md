# People Flow Analytics with YOLO26 and Multi-Object Tracking

基于 YOLO26 与多目标跟踪的固定摄像头人流分析系统。

## 中文

### 当前状态：V2 Phase 5

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
- V1 历史代码与数据来源说明的 `legacy/` 归档。

ROI、停留时间和 Ground Truth 计数评测仍按后续阶段实施。当前样例没有 Ground Truth，因此 README 不声明计数准确率。

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

输出包括：

```text
runs/demo/
├── annotated.mp4
├── tracks.csv
├── events.csv                 # 仅启用计数时生成
├── evidence/                  # 每个计数事件的证据帧
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

### Status: V2 Phase 5

This branch contains an installable `src`-layout application, the person detection and tracking pipeline, strict MOT17 support, and directional line-crossing counts based on persistent Track IDs and bottom-center foot points. Count events include CSV evidence records and JPEG evidence frames.

ROI occupancy, dwell time, and Ground Truth evaluation remain intentionally deferred. The bundled sample has no Ground Truth, so no counting-accuracy claim is made.

### Quick start

```bash
python -m venv .venv
pip install -e ".[dev]"
people-flow validate-config --config configs/base.yaml
python scripts/download_phase3_assets.py
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device cpu --output-dir runs/demo
people-flow run --source data/samples/People-counting-compressed.mp4 --model yolo26n.pt --tracker bytetrack.yaml --classes 0 --device 0 --output-dir runs/counting_demo --counting-line 100 500 1100 500 --enter-side positive
python scripts/prepare_mot17.py --root data/raw/MOT17 --split train
python scripts/mot_sequence_to_video.py --sequence data/raw/MOT17/train/MOT17-04-SDP --output data/interim/MOT17-04-SDP.mp4
ruff check .
pytest -q
python -m mypy src tests
```

The frozen Rock-Paper-Scissors V1 is available at `v1.0.0-rps-yolov8`. Dataset files, generated runs, and weights remain recoverable from that tag but are excluded from the V2 tree.

## License

The repository currently retains its MIT project license. Ultralytics code and models are offered under AGPL-3.0 and an Enterprise license; deploying the future web application must follow the selected Ultralytics license. The repository license does not override dependency, model, video, or dataset terms. See the official [Ultralytics licensing page](https://www.ultralytics.com/license) and the V1 audit in `docs/audit/V1_REPOSITORY_AUDIT.md`.
