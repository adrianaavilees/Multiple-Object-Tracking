import cv2
from ultralytics import YOLO
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIDEOS_PATH = r"C:\Users\simpl\Desktop\UNIVERSIDAD\cuarto\Métodes avançats PSIV (1er Semestre)\Proyecto detección\Videos\\"
SHOW_VIDEO = False  # Cambiar a True si quieres visualizar el video

# Línea de conteo
LINE_START = (10, 650)
LINE_END = (600, 650)
LINE_Y = 650

# ==========================================================
# CONFIGURACIÓN DE VIDEOS Y GROUND TRUTH
# ==========================================================
GROUND_TRUTH = {
    "output7.mp4": {"up": 6, "down": 2},
    "output2.mp4": {"up": 5, "down": 7},
    "output3.mp4": {"up": 3, "down": 10},
    "output5.mp4": {"up": 8, "down": 24},
}


# ==========================================================
# FUNCIONES AUXILIARES
# ==========================================================
def count_cars_across_line(tracked_objects, position_history, line_y, counted_ids_up, counted_ids_down):
    """Cuenta vehículos que cruzan la línea horizontal"""
    up_increment = 0
    down_increment = 0

    for obj_id, (cx, cy) in tracked_objects.items():
        if obj_id in position_history:
            prev_cy = position_history[obj_id]

            # Movimiento descendente
            if prev_cy < line_y <= cy and obj_id not in counted_ids_down:
                down_increment += 1
                counted_ids_down.add(obj_id)

            # Movimiento ascendente
            elif prev_cy > line_y >= cy and obj_id not in counted_ids_up:
                up_increment += 1
                counted_ids_up.add(obj_id)

        # Actualizar posición
        position_history[obj_id] = cy

    return up_increment, down_increment


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


class FileVideoReader:
    def __init__(self, video_name):
        self.video_path = VIDEOS_PATH + video_name
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error abriendo el archivo de video: {self.video_path}")

    def read(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Cargando modelo YOLO...")
        self.model = YOLO(model_path)
        print("Modelo cargado.")

    def detect(self, frame):
        results = self.model(frame, classes=2, verbose=False)  # Clase 2 = coches
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                bounding_boxes.append(box.xyxy[0].tolist())
        return bounding_boxes


# ==========================================================
# PROCESAMIENTO PRINCIPAL
# ==========================================================
def process_video(video_name, model_path="yolov8n.pt", show_video=False):
    """Procesa un video y devuelve los conteos detectados"""
    video_reader = FileVideoReader(video_name)
    detector = ObjectDetector(model_path)
    tracker = CentroidTracker(maxDisappeared=20)

    vehicles_up = 0
    vehicles_down = 0
    position_history = {}
    counted_ids_up = set()
    counted_ids_down = set()

    for frame in video_reader.read():
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        up_inc, down_inc = count_cars_across_line(
            tracked_objects, position_history, LINE_Y, counted_ids_up, counted_ids_down
        )
        vehicles_up += up_inc
        vehicles_down += down_inc

        if show_video:
            cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)
            for (objID, centroid) in tracked_objects.items():
                cv2.putText(frame, f"ID {objID}", (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imshow('Tracker', frame)
            if cv2.waitKey(1) == 27:
                break

    video_reader.release()
    return vehicles_up, vehicles_down


# ==========================================================
# COMPARACIÓN CON GROUND TRUTH Y MÉTRICAS
# ==========================================================
if __name__ == "__main__":
    results = []

    for video_name, truth in GROUND_TRUTH.items():
        print(f"\n=== Procesando {video_name} ===")
        detected_up, detected_down = process_video(video_name, show_video=SHOW_VIDEO)

        results.append({
            "video": video_name,
            "theoretical_up": truth["up"],
            "theoretical_down": truth["down"],
            "detected_up": detected_up,
            "detected_down": detected_down,
        })

        print(f"→ Detectados UP: {detected_up}, DOWN: {detected_down}")

    # Crear DataFrame
    df = pd.DataFrame(results)
    df["error_up"] = df["detected_up"] - df["theoretical_up"]
    df["error_down"] = df["detected_down"] - df["theoretical_down"]

    print("\n===== RESULTADOS =====")
    print(df)

    # Guardar en CSV
    df.to_csv("car_count_comparison.csv", index=False)
    print("\nResultados guardados en 'car_count_comparison.csv'")

    # Métricas
    df["abs_error_up"] = df["error_up"].abs()
    df["abs_error_down"] = df["error_down"].abs()
    MAE_up = df["abs_error_up"].mean()
    MAE_down = df["abs_error_down"].mean()

    print(f"\nMAE UP: {MAE_up:.2f}, MAE DOWN: {MAE_down:.2f}")

    # ======================================================
    # Gráficas comparativas
    # ======================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df["video"], df["theoretical_up"], label="Teórico UP", alpha=0.6)
    axes[0].bar(df["video"], df["detected_up"], label="Detectado UP", alpha=0.6)
    axes[0].set_title("Comparación - Vehículos UP")
    axes[0].legend()

    axes[1].bar(df["video"], df["theoretical_down"], label="Teórico DOWN", alpha=0.6)
    axes[1].bar(df["video"], df["detected_down"], label="Detectado DOWN", alpha=0.6)
    axes[1].set_title("Comparación - Vehículos DOWN")
    axes[1].legend()

    plt.show()

    # Error por video
    plt.figure(figsize=(8, 4))
    plt.bar(df["video"], df["error_up"], label="Error UP")
    plt.bar(df["video"], df["error_down"], label="Error DOWN", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Error de conteo (Detectado - Teórico)")
    plt.legend()
    plt.show()
