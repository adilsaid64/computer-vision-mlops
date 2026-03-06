import cv2
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


MODEL_PATH = "yolov8n.pt"

INFERENCE_FPS = 10
RECONNECT_DELAY = 5
MAX_FRAME_AGE = 2.0  # seconds before a frame is considered stale

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


RTSP_URLS: dict[str, str] = {
    "camera1": "rtsp://localhost:8554/camera1",
    "camera2": "rtsp://localhost:8554/camera2",
}


Frame = np.ndarray


@dataclass
class FramePacket:
    frame: Frame
    timestamp: float


class FrameStore:
    """
    Thread-safe latest-frame storage for all streams.
    """

    def __init__(self, stream_ids: list[str]):

        self._frames: dict[str, FramePacket | None] = {
            stream_id: None for stream_id in stream_ids
        }

        self._lock = threading.Lock()

    def update(self, stream_id: str, packet: FramePacket) -> None:

        with self._lock:
            self._frames[stream_id] = packet

    def get_batch(self):

        frames: list[Frame] = []
        stream_ids: list[str] = []
        timestamps: list[float] = []

        with self._lock:

            for stream_id, packet in self._frames.items():

                if packet is None:
                    continue

                frames.append(packet.frame)
                stream_ids.append(stream_id)
                timestamps.append(packet.timestamp)

        return stream_ids, frames, timestamps


class RTSPWorker:

    def __init__(self, stream_id: str, url: str, frame_store: FrameStore):

        self.stream_id = stream_id
        self.url = url
        self.frame_store = frame_store

        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> None:

        while True:

            logger.info(f"{self.stream_id}: connecting")

            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

            if cap.isOpened():

                self.cap = cap
                logger.info(f"{self.stream_id}: connected")

                return

            logger.warning(f"{self.stream_id}: connection failed")
            time.sleep(RECONNECT_DELAY)

    def run(self) -> None:

        self.connect()

        while True:

            if self.cap is None:
                self.connect()
                continue

            ret, frame = self.cap.read()

            if not ret:
                logger.warning(f"{self.stream_id}: frame read failed")
                self.connect()
                continue

            packet = FramePacket(
                frame=frame,
                timestamp=time.monotonic(),
            )

            self.frame_store.update(self.stream_id, packet)


class InferenceEngine:

    def __init__(
        self,
        frame_store: FrameStore,
        model_path: str,
        inference_fps: int,
    ):

        self.frame_store = frame_store
        self.model = YOLO(model_path)

        self.interval = 1 / inference_fps

    def process_results(self, stream_ids, results, timestamps) -> None:

        now = time.monotonic()

        for stream_id, result, ts in zip(stream_ids, results, timestamps):

            frame_age = now - ts

            if frame_age > MAX_FRAME_AGE:
                logger.warning(
                    f"{stream_id}: skipping stale frame age={frame_age:.2f}s"
                )
                continue

            vehicle_count = 0

            for box in result.boxes:

                cls_name = self.model.names[int(box.cls[0])]

                if cls_name in VEHICLE_CLASSES:
                    vehicle_count += 1

            logger.info(
                f"{stream_id}: {vehicle_count} vehicles | latency={frame_age:.2f}s"
            )

    def run(self) -> None:

        logger.info("Inference engine started")

        while True:

            start = time.monotonic()

            stream_ids, frames, timestamps = self.frame_store.get_batch()

            if frames:

                results = self.model(frames, verbose=False)

                self.process_results(stream_ids, results, timestamps)

            elapsed = time.monotonic() - start

            time.sleep(max(0, self.interval - elapsed))


class ObjectCounterService:

    def __init__(self):

        self.frame_store = FrameStore(list(RTSP_URLS.keys()))

        self.inference_engine = InferenceEngine(
            frame_store=self.frame_store,
            model_path=MODEL_PATH,
            inference_fps=INFERENCE_FPS,
        )

        self.executor = ThreadPoolExecutor(
            max_workers=len(RTSP_URLS)
        )

    def start_capture_workers(self) -> None:

        logger.info("Starting capture workers")

        for stream_id, url in RTSP_URLS.items():

            worker = RTSPWorker(
                stream_id=stream_id,
                url=url,
                frame_store=self.frame_store,
            )

            self.executor.submit(worker.run)

    def start_inference(self) -> None:

        thread = threading.Thread(
            target=self.inference_engine.run,
            daemon=True,
        )

        thread.start()

    def run(self) -> None:

        self.start_capture_workers()
        self.start_inference()

        while True:
            time.sleep(1)


def main() -> None:

    service = ObjectCounterService()
    service.run()


if __name__ == "__main__":
    main()