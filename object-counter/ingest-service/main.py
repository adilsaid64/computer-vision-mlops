import cv2
import time
import logging
import numpy as np
from collections import deque
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

RTSP_URLS = {"camera1": "rtsp://localhost:8554/camera1"}

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

if __name__ == "__main__":
    
    # Testing out how this woould work, so there is one main process
    # but each rtsp client gets its own thread
    for stream_id, url in RTSP_URLS.items():
        thread = threading.Thread(
            target=capture_loop,
            args=(stream_id, url),
            daemon=True,
        )
        thread.start()