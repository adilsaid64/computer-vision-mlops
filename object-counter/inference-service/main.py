from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    data = await file.read()
    img = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(img, cv2.IMREAD_COLOR)

    results = model(frame, verbose=False)[0]

    count = 0
    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if cls_name in VEHICLE_CLASSES:
            count += 1
            detections.append({
                "class": cls_name,
                "confidence": float(box.conf[0])
            })

    return {
        "vehicle_count": count,
        "detections": detections
    }

