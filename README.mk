# Rock-Paper-Scissors Detection ¡¤ YOLOv8

?? A lightweight, real-time detector that recognizes **Rock ¡¤ Paper ¡¤ Scissors** hand gestures using Ultralytics YOLOv8-nano.  
Trained on a custom dataset (mostly left-hand images) and reaches **mAP50 ¡Ö 0.93** after 50 epochs.

<div align="center">
  <img src="results/images/sample_detection.jpg" width="55%">
</div>

## Highlights
- **End-to-end pipeline** ¡ª data prep ¡ú training ¡ú validation ¡ú webcam demo.
- **Tiny model** (`best.pt` < 10 MB) runs at 30 FPS+ on laptop GPUs.
- **Plug-and-play scripts**: `train.py`, `predict.py`, `realtime_cam.py`, `evaluate.py`.
- Clear folder layout and documented hyper-parameters for easy reproduction.

> **Known limitation:** current dataset is left-hand dominant, so right-hand detection is less accurate. Future work: add balanced right-hand samples and fine-tune.

---
