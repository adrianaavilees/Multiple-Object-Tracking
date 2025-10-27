import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np

VIDEOS_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte2/Videos/"


class FileVideoReader:
    def __init__(self, video_name):
        self.video_path = VIDEOS_PATH + video_name
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

    def read(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()