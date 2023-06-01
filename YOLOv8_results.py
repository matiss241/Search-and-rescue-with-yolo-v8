# Record a vide with web cam and make predictions with YOLO

from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/best.pt")

model.predict(source="0", show=True, conf=0.5)