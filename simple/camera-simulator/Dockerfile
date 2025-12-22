FROM jrottenberg/ffmpeg:6.0-alpine

WORKDIR /app
COPY videos /videos

CMD [
  "-re",
  "-stream_loop", "-1",
  "-i", "/videos/traffic.mp4",
  "-c", "copy",
  "-f", "rtsp",
  "rtsp://rtsp-server:8554/camera1"
]

