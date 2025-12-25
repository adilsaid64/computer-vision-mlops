import cv2
from fastapi import FastAPI
from fastapi.responses import Response
import requests
import time

app = FastAPI()

RTSP_URL = "rtsp://localhost:8554/camera1?rtsp_transport=tcp"
INFER_URL = "http://localhost:8000/infer"

class RTSPClient:
    def __init__(self, rtsp_url:str, wait_time:float):
        self.rtsp_url = rtsp_url
        self.wait_time = wait_time
        self.cap = None

    def connect(self):
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if self.cap.isOpened():
            print('connected')
        else:
            raise RuntimeError('can not connected to rtsp stream')

    def get_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            sucess,encoded = cv2.imencode('.jpg', frame)
            return encoded.tobytes()
        else:
            return None


def get_inference(jpeg_bytes: bytes) -> dict[str, int]:
    files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
    r = requests.post(INFER_URL, files=files, timeout=5)
    return r.json()


rtsp_client = RTSPClient(rtsp_url=RTSP_URL, wait_time=1)
rtsp_client.connect()

while True:
    jpeg_bytes = rtsp_client.get_frame()
    r = get_inference(jpeg_bytes)
    print(r)
    time.sleep(1)
