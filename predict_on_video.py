# Predict with YOLO on all videos in directory
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
import torch
import os
torch.cuda.is_available()
model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/training/Optimizers/SGD/weights/best.pt")
model.to('cuda')
model.predict('C:/Users/MC00202/Desktop/YOLOv8/assets/SAR/1692533987230.mp4', save=True, imgsz=640, conf=0.5)