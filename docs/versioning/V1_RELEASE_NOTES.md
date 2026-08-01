# V1.0.0 — YOLOv8 Rock-Paper-Scissors Detection

## 中文说明

### 项目用途

V1 是一个基于 Ultralytics YOLOv8n 的石头、剪刀、布手势目标检测学习项目，保留了从训练、验证到图片推理和实时摄像头演示的早期实践代码。

### 数据集来源

仓库中的数据来自 Roboflow Universe 的 [Rock Paper Scissors SXSW](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw) 数据集导出。仓库内的 Roboflow 元数据说明该导出版本包含 7,521 张 YOLOv8 格式图像，但同时将许可证标记为 `undefined`。因此，使用或再分发数据前应自行核实原始数据集及其组成来源的许可条款。

### 现有脚本

- `scripts/train.py`：使用 `yolov8n.pt` 和 `data.yaml` 训练 50 个 epoch。
- `scripts/推理测试.py`：对测试图片目录执行批量推理并保存标签和置信度。
- `scripts/实时摄像头测试.py`：使用默认摄像头进行实时检测。
- `scripts/评估模型.py`：在验证集上调用 Ultralytics 模型评估。

### 已知限制

- `data.yaml`、批量推理、摄像头推理和评估脚本包含 `D:/Pythonmodel/...` 等机器相关绝对路径，换机后不能直接复现。
- README 列出了 `requirements.txt`，但当前提交中不存在该文件，依赖版本未被锁定。
- 部分脚本在导入时立即加载模型或访问摄像头，没有命令行参数、配置校验和自动化测试。
- 评估脚本包含针对 Windows 多进程的设置；整体跨平台行为未经验证。
- README 提到数据以左手样本为主，右手检测能力有限。
- 仓库包含完整数据集、训练输出和多份模型权重，体积较大。
- 本 Release 不声明任何新的性能指标；历史 README 和训练产物中的数值未在本次冻结过程中重新运行验证。

### 冻结说明

V1 作为 YOLOv8 石头剪刀布历史学习版本冻结。后续“基于 YOLO26 与多目标跟踪的人流分析系统”开发只在 `v2/yolo26-people-flow` 分支进行，不直接改写 `main` 或此 V1 标签的历史。

## English Notes

### Purpose

V1 is a learning project for Rock-Paper-Scissors object detection with Ultralytics YOLOv8n. It preserves the original training, validation, batch-image inference, and webcam demonstration scripts.

### Dataset source

The checked-in data was exported from the [Rock Paper Scissors SXSW](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw) dataset on Roboflow Universe. Its bundled metadata states that the export contains 7,521 images in YOLOv8 format, while also declaring the license as `undefined`. Users must verify the licensing terms of the source dataset and its aggregated sources before use or redistribution.

### Included scripts

- `scripts/train.py`: trains `yolov8n.pt` for 50 epochs using `data.yaml`.
- `scripts/推理测试.py`: runs batch inference on a test-image directory.
- `scripts/实时摄像头测试.py`: runs detection from the default webcam.
- `scripts/评估模型.py`: invokes Ultralytics validation on the configured dataset.

### Known limitations

- Dataset and model locations are hard-coded with machine-specific `D:/Pythonmodel/...` paths.
- The README references `requirements.txt`, but that file is absent from the frozen commit and dependency versions are not pinned.
- Some scripts load models or access the webcam at import time and provide no CLI, configuration validation, or automated tests.
- Cross-platform behavior is unverified, and the evaluation script contains Windows-specific multiprocessing handling.
- The README describes a left-hand-dominant dataset and weaker right-hand detection.
- The repository includes the full dataset, generated training outputs, and duplicate model weights.
- This Release makes no new performance claim. Values found in the historical README or generated training artifacts were not independently reproduced during the freeze.

### Freeze policy

V1 is frozen as the historical YOLOv8 Rock-Paper-Scissors learning version. Development of the YOLO26 and multi-object-tracking people-flow system will occur only on `v2/yolo26-people-flow`; the `main` branch and this V1 tag will not be rewritten.
