import cv2

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

    def show(self):
        for frame in self.read():
            cv2.imshow('Video', frame)
            key = cv2.waitKey(1) 

            # Close the video window if 'Esc' is pressed or the window is closed
            if key == 27 or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                break

        print("Closing video...")
        self.cap.release()
        cv2.destroyAllWindows()

#TODO: añadir la detección, el seguimiento y el conteo de coches...

######## MAIN ########
video_name = "video1.mp4"
fileVideoReader = FileVideoReader(video_name)
fileVideoReader.show()