from vehicle_counter import VehicleCounter


if __name__ == "__main__":
    counter = VehicleCounter("yolov8n.pt", tracker="complex")
    # Procesar un solo video
    counter.process_video("video1.mp4", use_roi=True, show_video=False)