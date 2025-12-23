import cv2
from fastapi import FastAPI
from fastapi.responses import Response
app = FastAPI()

RTSP_URL = "rtsp://localhost:8554/camera1?rtsp_transport=tcp"

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

rtsp_client = RTSPClient(rtsp_url=RTSP_URL, wait_time=1)
rtsp_client.connect()

@app.get('/frame')
def get_frame():
    return Response(content = rtsp_client.get_frame(), media_type='image/jpeg')
