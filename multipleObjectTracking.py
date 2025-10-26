import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np

VIDEOS_PATH = "videos/"

# DEFINE THE COORDINATES FOR THE LINE FOR COUNTING
LINE_START = (10, 650)
LINE_END = (600, 650)

class FileVideoReader:
    def __init__(self, video_name):
        self.video_path = VIDEOS_PATH + video_name
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {self.video_path}")

    def read(self):
        while True:
            # cap.read() reads the next frame from the video
            # ret indicates if the frame was read successfully (TRUE)
            # frame is the image 
            ret, frame = self.cap.read()
            
            # If ret=False, it means the video has ended
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect(self, frame):
        results = self.model(frame, classes=2, verbose=False)  # Class 2 is for cars

        # * THIS IMPLEMENTATION IS FOR THE OPENCV TRACKER
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                bounding_boxes.append(box.xyxy[0].tolist())
        
        return bounding_boxes

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        
    def register(self, centroid):
        # when registering an object, we use the next available object ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        # to deregister an object ID, we delete the object ID from both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, bounding_boxes):
        if len(bounding_boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of conscutive frames where an object has been marked as missing, desregistered it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
            return self.objects

        inputCentroids = np.zeros((len(bounding_boxes), 2), dtype="int")

        # Loop over the bounding boxes rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(bounding_boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        # If we are not tracking any object, register all new ones
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # If we are already tracking an object, associate it
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            
            # Find the smallest value in each row and column
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Associate the IDs with the new centroids
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            # Check with objects have not been associate
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If there's more existent objects than new ones, we mark them as desapeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # If there's more new objects than existing ones, we registered them
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


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


def process_video(video_name, show_video): #* Change this depending if we want to visualize the video or not
    LINE_Y = 650

    video_reader = FileVideoReader(video_name)
    detector = ObjectDetector("yolov8n.pt") #TODO: probar yolov8n.pt para más velocidad. El yolov8s va MUY lento
    tracker = CentroidTracker(maxDisappeared=20)

    vehicles_up, vehicles_down = 0, 0
    position_history = {}
    counted_ids_up, counted_ids_down = set(), set()

    for frame in video_reader.read():
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        up_inc, down_inc = count_cars_across_line(tracked_objects, position_history, LINE_Y, counted_ids_up, counted_ids_down)
        vehicles_up += up_inc
        vehicles_down += down_inc

        if show_video:
            cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3) # Draw the counting line    
            for (objID, centroid) in tracked_objects.items():
                cv2.putText(frame, f"ID {objID}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            cv2.putText(frame, f"UP: {vehicles_up}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"DOWN: {vehicles_down}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.imshow('Tracker', frame)

            key = cv2.waitKey(1) 
            # Close the video window if 'Esc' is pressed or the window is closed
            if key == 27 or cv2.getWindowProperty('Tracker', cv2.WND_PROP_VISIBLE) < 1:
                break
    
    video_reader.release()
    return vehicles_up, vehicles_down


######## MAIN ########
if __name__ == "__main__":
    video_file = "video1.mp4" 
    up_count, down_count = process_video(video_file, show_video=True)
    print(f"Total cars moving UP: {up_count}")
    print(f"Total cars moving DOWN: {down_count}")