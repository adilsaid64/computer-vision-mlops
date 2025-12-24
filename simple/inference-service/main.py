import requests
import numpy as np
import cv2
from ultralytics import YOLO

INGEST_URL = "http://localhost:8000/frame"

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def get_frame():
    r = requests.get(INGEST_URL, timeout=5)
    img = np.frombuffer(r.content, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def count_vehicles(frame):
    results = model(frame, verbose=False)[0]

    count = 0
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name in VEHICLE_CLASSES:
            count += 1

    return count


def main():
    frame = get_frame()
    count = count_vehicles(frame)
    print(f"Vehicles in frame: {count}")


if __name__ == "__main__":
    while True:
    main()

