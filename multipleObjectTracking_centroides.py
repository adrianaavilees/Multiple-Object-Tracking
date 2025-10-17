import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# -----------------------------
# Rutas y parámetros
# -----------------------------
VIDEOS_PATH = r"C:\Users\simpl\Desktop\UNIVERSIDAD\cuarto\Métodes avançats PSIV (1er Semestre)\Proyecto detección\Videos"
VIDEO_NAME = "output2.mp4"
OUTPUT_VIDEO = "output_tracked.mp4"

LINE_Y = 650  # posición de la línea horizontal
LINE_START = (10, LINE_Y)
LINE_END = (600, LINE_Y)

# -----------------------------
# Clase para leer vídeo
# -----------------------------
class FileVideoReader:
    def __init__(self, video_name):
        self.video_path = os.path.join(VIDEOS_PATH, video_name)
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

# -----------------------------
# Clase para detección con YOLO
# -----------------------------
class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("Model loaded.")

    def detect(self, frame):
        results = self.model(frame, classes=2, verbose=False)  # Clase 2 = coches
        final_cars_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width, height = x2 - x1, y2 - y1
                confidence = box.conf[0]
                class_id = box.cls[0]
                # Formato DeepSORT: [x, y, width, height], confianza, class_id
                final_cars_boxes.append(([x1, y1, width, height], confidence, int(class_id)))
        return final_cars_boxes

# -----------------------------
# Función para contar coches cruzando la línea
# -----------------------------
def count_cars_across_line(vehicle_positions, memory, line_y, vehicles_up, vehicles_down):
    for obj_id, (cx, cy) in vehicle_positions.items():
        if obj_id in memory:
            prev_cy = memory[obj_id]
            # Cruce hacia abajo
            if prev_cy < line_y <= cy:
                vehicles_down += 1
            # Cruce hacia arriba
            elif prev_cy > line_y >= cy:
                vehicles_up += 1
        memory[obj_id] = cy
    return vehicles_up, vehicles_down

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Inicialización
    video_reader = FileVideoReader(VIDEO_NAME)
    detector = ObjectDetector("yolov8n.pt")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

    # Contadores y memoria
    vehicles_up = 0
    vehicles_down = 0
    memory = {}

    # Información de salida
    first_frame = next(video_reader.read())
    height, width = first_frame.shape[:2]
    fps = 30  # Ajusta si quieres usar el FPS real del vídeo
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Volvemos a iterar desde el primer frame
    video_reader.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for frame in video_reader.read():
        # Línea de conteo
        cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)

        # Detección y tracking
        detections = detector.detect(frame)
        tracked_objects = tracker.update_tracks(detections, frame=frame)

        vehicle_positions = {}

        for obj in tracked_objects:
            if not obj.is_confirmed():
                continue
            obj_id = obj.track_id
            x1, y1, x2, y2 = map(int, obj.to_ltrb())
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            vehicle_positions[obj_id] = (cx, cy)

            # Dibujar caja y centroide
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Contar coches
        vehicles_up, vehicles_down = count_cars_across_line(vehicle_positions, memory, LINE_Y, vehicles_up, vehicles_down)

        # Mostrar contadores
        cv2.putText(frame, f"Up: {vehicles_up}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Down: {vehicles_down}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Mostrar frame y guardar
        cv2.imshow("Detections", frame)
        out.write(frame)

        key = cv2.waitKey(1)
        if key == 27 or cv2.getWindowProperty('Detections', cv2.WND_PROP_VISIBLE) < 1:
            break

    print("✅ Video procesado. Resultado guardado en:", OUTPUT_VIDEO)
    video_reader.release()
    out.release()

