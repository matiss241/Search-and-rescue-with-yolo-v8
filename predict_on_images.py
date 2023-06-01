# Predict with YOLO on all images in assets dir.
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import os

model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/best.pt")
assets_dir = 'C:/Users/MC00202/Desktop/YOLOv8/assets'
file_list = os.listdir(assets_dir)

for file_name in file_list:
  file = os.path.join(assets_dir, file_name)

  if os.path.isfile(file):
    result = model.predict(file, save=True, imgsz=640, conf=0.5);
