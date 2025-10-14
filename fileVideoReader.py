import cv2
from ultralytics import YOLO

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

        final_cars_boxes = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                width, height = x2 - x1, y2 - y1
                final_cars_boxes.append([x1, y1, width, height])

        return final_cars_boxes



#TODO: añadir la detección, el seguimiento y el conteo de coches...

######## MAIN ########
if __name__ == "__main__":
    video_name = "video1.mp4"
    video_reader = FileVideoReader(video_name)
    detector = ObjectDetector("yolov8n.pt") #TODO: probar yolov8n.pt para más velocidad. El yolov8s va MUY lento

    for frame in video_reader.read():
        cars_boxes = detector.detect(frame)
        
        for box in cars_boxes:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # blue color box for cars
            cv2.putText(frame, "Car", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) 
        
        cv2.imshow('Detections', frame)

        key = cv2.waitKey(1) 
        # Close the video window if 'Esc' is pressed or the window is closed
        if key == 27 or cv2.getWindowProperty('Detections', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    print("Closing video...")
    video_reader.release()