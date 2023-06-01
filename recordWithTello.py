# Filming with Tello
# References:
# https://github.com/damiafuentes/DJITelloPy/blob/master/examples/manual-control-opencv.py
import numpy as np
import time
import cv2
from djitellopy import Tello
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/best.pt")

tello = Tello()

tello.connect()
tello.streamon()


while True:
    frame_read = tello.get_frame_read().frame

    model.predict(frame_read, show=True, save=True, conf=0.5)

    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break

cv2.destroyAllWindows()
tello.streamoff()
tello.end()