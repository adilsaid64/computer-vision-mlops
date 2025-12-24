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

while True:
    frame = get_frame()

    results = model(frame, verbose=False)[0]

    count = 0

    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name not in VEHICLE_CLASSES:
            continue

        count += 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            cls_name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1
        )

    cv2.putText(
        frame,
        f"Vehicles in frame: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()

