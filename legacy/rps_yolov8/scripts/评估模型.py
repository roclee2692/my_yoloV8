# 评估模型.py

import multiprocessing
from ultralytics import YOLO

def main():
    # 1. 指定模型权重路径
    weights_path = r'D:/Pythonmodel/my_yolo_project/runs/detect/train/weights/best.pt'
    # 2. 指定 data.yaml 配置文件路径
    data_yaml   = r'D:/Pythonmodel/my_yolo_project/data.yaml'

    # 3. 加载训练好的模型
    model = YOLO(weights_path)

    # 4. 在验证集上评估，workers=0 避免 Windows 多进程冲突
    metrics = model.val(data=data_yaml, workers=0)

    # 5. 打印评估结果
    print("===== 验证集评估结果 =====")
    print(metrics)

if __name__ == '__main__':
    # Windows 下多进程支持
    multiprocessing.freeze_support()
    main()
