# Multiple-Object-Tracking
This project uses the YOLOv8 object detection model and a custom Centroid Tracker to detect, track, and count vehicles in a video stream. It is designed to monitor a parking garage ramp, counting vehicles as they move "UP" (exiting) or "DOWN" (entering) to estimate occupancy.

# Features
    -  Object Detection: Uses YOLOv8 (specifically yolov8n.pt) to detect cars in real-time.

    - Multi-Object Tracking: Implements a simple and efficient CentroidTracker to assign a unique ID to each detected vehicle and track its movement across frames.

    - Directional Counting: Logs when a tracked object crosses a predefined horizontal line, incrementing separate counters for "UP" and "DOWN" movements.

    - Visualization: Displays the video feed with the counting line, tracked object centroids, object IDs, and a live scoreboard of "UP" and "DOWN" counts.

# How It Works
The processing pipeline is structured as follows:
1. Video Reading (FileVideoReader): A class reads the video file frame by frame.

2. Detection (ObjectDetector): Each frame is passed to the YOLOv8 model. The model returns a list of bounding boxes for all detected 'car' objects (class 2).

3. Tracking (CentroidTracker):
    - The bounding boxes are used to calculate the centroid (center point) of each detected car.
    - The tracker associates these new centroids with existing tracked objects by calculating the Euclidean distance.
    - If a centroid is new, it's registered with a unique ID.
    - If an existing object is not detected for a specified number of frames (maxDisappeared), it is deregistered.

4. Counting (count_cars_across_line):
    - This function checks the position of each tracked object relative to a horizontal line (LINE_Y).
    - It compares the object's previous Y-coordinate with its current Y-coordinate.
    - If the object crosses the line (prev_cy < line_y <= cy), it's counted as "DOWN".
    - If it crosses in the other direction (prev_cy > line_y >= cy), it's counted as "UP".
    - A set is used to ensure each vehicle ID is counted only once per direction.

5. Visualization (process_video):
    - OpenCV (cv2) is used to draw the tracking information (centroids, IDs) and the counting line and scoreboard onto the frame.
    - The processed video is displayed on the screen.


# References
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
