import cv2
import time
import logging
import numpy as np
from collections import deque
import threading
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

RTSP_URLS = {
    "camera1": "rtsp://localhost:8554/camera1", 
    "camera2": "rtsp://localhost:8554/camera2",
}

stream_buffers = {
    stream_id: deque(maxlen=1)
    for stream_id in RTSP_URLS
}

class RTSPClient:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None

    def connect(self) -> None:
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot connect to RTSP stream")
        logger.info("Connected")

    def get_frame(self) -> np.ndarray | None:
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return None

        return frame


def capture_loop(stream_id: str, rtsp_url: str):
    client = RTSPClient(rtsp_url)
    client.connect()

    while True:
        frame = client.get_frame()
        if frame is None:
            continue
        stream_buffers[stream_id].append(frame)


def inference_loop():
    logger.info("Inference thread started")

    while True:
        frames = []
        stream_ids = []

        for stream_id, buffer in stream_buffers.items():
            if buffer:
                frames.append(buffer[-1])
                stream_ids.append(stream_id)

        if not frames:
            time.sleep(0.01)
            continue

        results = model(frames, verbose=False)

        # Process results per stream
        for stream_id, result in zip(stream_ids, results):
            count = 0
            detections = []

            for box in result.boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name in VEHICLE_CLASSES:
                    count += 1
                    detections.append({
                        "class": cls_name,
                        "confidence": float(box.conf[0])
                    })

            logger.info(f"{stream_id}: {count} vehicles")


if __name__ == "__main__":
    
    for stream_id, url in RTSP_URLS.items():
        thread = threading.Thread(
            target=capture_loop,
            args=(stream_id, url),
            daemon=True,
        )
        thread.start()

    infer_thread = threading.Thread(
        target=inference_loop,
        daemon=True,
    )
    infer_thread.start()


    while True:
        time.sleep(1)