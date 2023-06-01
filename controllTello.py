# Controlling tello with wasd
# References:
# https://github.com/damiafuentes/DJITelloPy/blob/master/examples/manual-control-opencv.py
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from djitellopy import Tello
import cv2, math, time

tello = Tello()
tello.connect()
model = YOLO("C:/Users/MC00202/Desktop/YOLOv8/best.pt")

tello.streamon()
frame_read = tello.get_frame_read()
tello.takeoff()

while True:
    img = frame_read.frame
    # cv2.imshow("drone", img)
    model.predict(img, show=True, save=True, conf=0.5)

    key = cv2.waitKey(1) & 0xff
    if key == 27: # ESC
        break
    elif key == ord('w'):
        tello.move_forward(30)
    elif key == ord('s'):
        tello.move_back(30)
    elif key == ord('a'):
        tello.move_left(30)
    elif key == ord('d'):
        tello.move_right(30)
    elif key == ord('e'):
        tello.rotate_clockwise(30)
    elif key == ord('q'):
        tello.rotate_counter_clockwise(30)
    elif key == ord('r'):
        tello.move_up(30)
    elif key == ord('f'):
        tello.move_down(30)

tello.land()