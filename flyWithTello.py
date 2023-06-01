# Script for recording a video with Tello
# References:
# https://github.com/damiafuentes/DJITelloPy/blob/master/examples/record-video.py

import numpy as np
import time
import cv2
from threading import Thread
from djitellopy import Tello
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/best.pt")
keepRecording = True
tello = Tello()

tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()


def videoRecorder():
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keepRecording:
        video.write(frame_read.frame)
        model.predict(frame_read.frame, show=True, save=True, conf=0.5)
        time.sleep(1 / 30)

    video.release()

recorder = Thread(target=videoRecorder)
recorder.start()


tello.takeoff()
tello.move_forward(10);
tello.move_up(100)
time.sleep(30)
tello.rotate_clockwise(50)
time.sleep(30);
tello.rotate_counter_clockwise(50)
time.sleep(30);
tello.land()

keepRecording = False
recorder.join()