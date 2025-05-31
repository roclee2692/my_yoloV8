from ultralytics import YOLO
import os

# 加载训练好的模型
model = YOLO('D:/Pythonmodel/my_yolo_project/runs/detect/train/weights/best.pt')

# 指定测试图片目录
test_dir = r'D:\Pythonmodel\my_yolo_project\datasets\test\images'

# 批量推理
results = model.predict(source=test_dir, save=True, save_txt=True, save_conf=True)

print("推理完成，结果已保存到 runs/predict/ 下！")
