import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect(self, frame):
        results = self.model(frame, classes=2, verbose=False)  # Class 2 = cars
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                bounding_boxes.append(box.xyxy[0].tolist())
        return bounding_boxes
