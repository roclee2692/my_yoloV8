if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=50, imgsz=640)
