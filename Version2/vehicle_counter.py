import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np

from detector import ObjectDetector
from trackers import CentroidTracker, ComplexTracker
from video_reader import FileVideoReader

#VIDEOS_PATH = "videos/"
VIDEOS_PATH = "/Users/lianbaguebatlle/Desktop/Dades/Tercer/1rsemestre/PSIV2/Projecte2/Videos/"

# Parámetros globales por defecto
ROI_Y1 = 400
ROI_Y2 = 720
LINE_START = (10, 650)
LINE_END = (600, 650)
LINE_Y = 650


def count_cars_across_line(tracked_objects, position_history, line_y, counted_ids_up, counted_ids_down):
    """Counts vehicles crossing a horizontal line"""
    up_increment = 0
    down_increment = 0

    for obj_id, (cx, cy) in tracked_objects.items():
        if obj_id in position_history:
            prev_cy = position_history[obj_id]

            # Moving down
            if prev_cy < line_y <= cy and obj_id not in counted_ids_down:
                down_increment += 1
                counted_ids_down.add(obj_id)

            # Moving up
            elif prev_cy > line_y >= cy and obj_id not in counted_ids_up:
                up_increment += 1
                counted_ids_up.add(obj_id)

        # Update the actual position
        position_history[obj_id] = cy

    return up_increment, down_increment



# NUEVA CLASE PRINCIPAL
class VehicleCounter:
    def __init__(self, model_path="yolov8n.pt", tracker = "centroid"):
        if tracker == "centroid":
            self.tracker = CentroidTracker(maxDisappeared=20)
        else:
            self.tracker = ComplexTracker(maxDisappeared=20, maxDistance=50, direction=True, color=False, speed=False)
        self.detector = ObjectDetector(model_path)
        

    def process_video(self, video_name, use_roi=True, show_video=False):
        video_reader = FileVideoReader(video_name)

        vehicles_up = 0
        vehicles_down = 0
        position_history = {}
        counted_ids_up = set()
        counted_ids_down = set()

        for frame in video_reader.read():
            if use_roi:
                y_offset = ROI_Y1
                roi = frame[ROI_Y1:ROI_Y2, :]
                detections = self.detector.detect(roi)

                adjusted_detections = []
                for (x1, y1, x2, y2) in detections:
                    adjusted_detections.append([x1, y1 + y_offset, x2, y2 + y_offset])
                detections = adjusted_detections
            else:
                detections = self.detector.detect(frame)

            tracked_objects = self.tracker.update(detections)

            up_inc, down_inc = count_cars_across_line(
                tracked_objects, position_history, LINE_Y, counted_ids_up, counted_ids_down
            )
            vehicles_up += up_inc
            vehicles_down += down_inc

            if show_video:
                cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)
                for (objID, centroid) in tracked_objects.items():
                    cv2.putText(
                        frame,
                        f"ID {objID}",
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                cv2.putText(frame, f"UP: {vehicles_up}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"DOWN: {vehicles_down}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                cv2.imshow("Tracker", frame)
                key = cv2.waitKey(1)
                if key == 27 or cv2.getWindowProperty("Tracker", cv2.WND_PROP_VISIBLE) < 1:
                    break

        video_reader.release()
        print(f"Video: {video_name} → UP: {vehicles_up}, DOWN: {vehicles_down}")
        return vehicles_up, vehicles_down



