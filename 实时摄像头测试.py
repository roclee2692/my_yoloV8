import cv2
from ultralytics import YOLO

model = YOLO('D:/Pythonmodel/my_yolo_project/runs/detect/train/weights/best.pt')

cap = cv2.VideoCapture(0)  # 0表示默认摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, show=True, conf=0.3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
