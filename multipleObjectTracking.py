import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np

VIDEOS_PATH = "videos/"

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