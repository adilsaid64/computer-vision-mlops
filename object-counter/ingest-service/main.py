import cv2
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

RTSP_URL = "rtsp://localhost:8554/camera1"

class RTSPClient:
    def __init__(self, rtsp_url: str):
        self.rtsp_url = rtsp_url
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot connect to RTSP stream")
        logger.info("Connected")

    def get_frame(self):
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame")
            return None

        return frame
    
if __name__ == "__main__":
    
    rtsp_client = RTSPClient(RTSP_URL)
    rtsp_client.connect()

    while True:
        frame = rtsp_client.get_frame()
        if frame is None:
            continue
        logger.info("Frame shape: %s", frame.shape)
        time.sleep(1)
