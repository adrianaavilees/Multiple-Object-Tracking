import cv2
from ultralytics import YOLO #! pip install ultralytics
from scipy.spatial import distance as dist #! pip install scipy
from collections import OrderedDict 
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, bounding_boxes):
        if len(bounding_boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(bounding_boxes), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(bounding_boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


class ComplexTracker(CentroidTracker):
    def __init__(self, maxDisappeared=50, maxDistance=50, direction=True, color=False, speed=False):
        super().__init__(maxDisappeared)
        self.maxDistance = maxDistance
        self.direction = direction
        self.color = color
        self.speed = speed
        self.prev_positions = OrderedDict()  # para dirección/oclusiones opcional

    def update(self, bounding_boxes):
        if len(bounding_boxes) == 0:
            # Mismo comportamiento que CentroidTracker
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(bounding_boxes), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(bounding_boxes):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
            return self.objects

        # Objetos existentes
        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        # Distancias
        D = dist.cdist(np.array(objectCentroids), inputCentroids)

        # Filtrar por maxDistance si no es infinito
        if self.maxDistance != float("inf"):
            D[D > self.maxDistance] = np.inf

        # Asignación: idéntica a CentroidTracker
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] == np.inf:
                continue

            objectID = objectIDs[row]
            new_centroid = inputCentroids[col]

            # Opcional: dirección
            if self.direction:
                prev_cx, prev_cy = self.objects[objectID]
                dx = new_centroid[0] - prev_cx
                dy = new_centroid[1] - prev_cy
                self.prev_positions[objectID] = {"dx": dx, "dy": dy}

            self.objects[objectID] = new_centroid
            self.disappeared[objectID] = 0

            usedRows.add(row)
            usedCols.add(col)

        # Objetos no asignados
        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        # Deregistrar o registrar igual que CentroidTracker
        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
        else:
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects
