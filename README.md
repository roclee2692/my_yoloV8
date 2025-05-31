# Rock-Paper-Scissors Detection · YOLOv8

🖐️ A lightweight, real-time detector that recognizes **Rock · Paper · Scissors** hand gestures using Ultralytics YOLOv8-nano.  
Trained on a custom dataset (mainly left-hand samples) and reaches **mAP50 ≈ 0.93** after 50 epochs.

<div align="center">
  <img src="results/images/sample_detection.jpg" width="55%">
</div>

## 📦 项目结构
本项目结构简要如下（部分文件省略，仅展示主要部分）：

my_yolov8/
├── datasets/ # 数据集文件夹
│ ├── Rock Paper Scissors SXSW/ # Roboflow 下载的原始数据集，含 README
│ ├── test/ # 测试集
│ ├── train/ # 训练集
│ └── valid/ # 验证集
│
├── results/ # 提取后的推理结果和训练指标图
│
├── runs/ # YOLO 自动生成的完整训练记录（可选上传）
│ └── detect/...
│
├── scripts/ # Python 脚本
│ ├── train.py # 模型训练脚本
│ ├── 推理测试.py # 批量图片推理脚本
│ ├── 实时摄像头测试.py # 实时检测脚本
│ └── 评估模型.py # 模型评估脚本
│
├── weights/ # 模型权重
│ └── best.pt # 训练出的最佳模型
│
├── data.yaml # YOLO 数据集配置文件
├── README.md # 项目说明文件
├── requirements.txt # Python 依赖
└── .gitignore # Git 忽略规则
> 💡 **完整训练结果位于 YOLO 自动生成的 `runs/` 文件夹，为简洁展示，仅提取部分结果到 `results/`。**

> 🔍 数据集原始文件夹 `Rock Paper Scissors SXSW` 来自 Roboflow，包含 README 文件说明来源。

## 🔥 Highlights
- **End-to-end pipeline** — data prep → training → validation → webcam demo.  
- **Tiny model** (`best.pt` < 10 MB) runs at 30 FPS + on laptop GPUs.  
- **Plug-and-play scripts**: `train.py`, `predict.py`, `realtime_cam.py`, `evaluate.py`.  
- Clear folder layout and documented hyper-parameters for easy reproduction.

> **Known limitation:** current dataset is left-hand dominant, so right-hand detection is less accurate. Future work: add balanced right-hand samples and fine-tune.
## 🚀 使用说明

### 训练模型
```bash
python scripts/train.py
批量推理图片
bash

python scripts/predict.py --source data/test/images
实时摄像头检测
bash

python scripts/realtime_cam.py
评估模型
bash

python scripts/evaluate.py
📊 结果展示
训练曲线
（完整曲线见 results/metrics/）


推理示例
<div align="center"> <img src="results/images/sample1.jpg" width="45%"> <img src="results/images/sample2.jpg" width="45%"> </div>
⚙️ 模型权重
best.pt — 训练后的最佳模型，已保存在 weights/。

如果模型较大，可上传至网盘并在此处提供下载链接：
Download best.pt (Google Drive)

🚀 导出模型（ONNX / TensorRT）
python

from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.export(format="onnx")      # or format="engine"
📌 已知问题 & 后续优化
数据集中左手样本较多，右手检测准确率较低。
后续可补充右手样本并再次 fine-tune。

摄像头推理略有卡顿，可尝试更轻量模型或降低分辨率。

未来计划支持更多手势类别（如 OK、比心 等）。

📄 License
This project is licensed under the MIT License.

🙌 特别感谢
Ultralytics YOLO — 强大的目标检测工具包

Roboflow — 数据集平台支持

